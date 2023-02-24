//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Transpose Convolution Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../common/logging.h"
#include "../../gl/glinfo.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/glexception.h"

#include "transconvlayerbase_vanilla.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
TransConvLayerBase::TransConvLayerBase(const ConvLayerBuilder & builder, int layerNumber) : ConvLayerBase(builder, layerNumber) {
    if (upsample_ != 2) THROW_EXCEPTION_ARGS(FynException, "Only stride 2 transpose conv layers are supported for now");
    if (inputPadding_) THROW_EXCEPTION_ARGS(FynException, "Currently no input padding supported / tested");
    // TODO (mw) implement support for that
    if (flags_ & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException, "No support for post-BN in transpose convolution as of now");
    assert(builder.downsample_[0] == builder.downsample_[1]);
    assert(builder.upsample_[0] == builder.upsample_[1]);
    assert(builder.downsample_[0] == 1);
    maxRenderTargets_ = GLInfo::getMaximumRecommendedDrawBuffers();
    int maxvecs = GLInfo::getMaxUniformVectors(GLInfo::FRAGMENT);
    int biasvec = (outputPadding_) ? 1 : 0;
    // -----------------------------------------------------------------
    // Limit number of render targets by overhead resulting from passing
    // variables from vertex to fragment shader...
    // -----------------------------------------------------------------
    int maxrt = std::max(1, (maxvecs - VEC_OVERHEAD) / (4*1+biasvec));
    maxRenderTargets_ = std::min(maxRenderTargets_, maxrt);
    upsample_ = builder.upsample_[0];
    viewport_[0] = width_*upsample_ + 2*outputPadding_;
    viewport_[1] = height_*upsample_ + 2*outputPadding_;
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
TransConvLayerBase::~TransConvLayerBase() {
    if (weights_) delete weights_;
    weights_ = nullptr;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void TransConvLayerBase::cleanup() {
    if (coordBuffer_) delete coordBuffer_;
    if (textureBuffer_) delete textureBuffer_;
    if (indexBuffer_) delete indexBuffer_;
    if (vertexArray_) delete vertexArray_;
    coordBuffer_ = nullptr;
    textureBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    if ((context_.isCurrent()) && (stencilBuffer_ != 0)) {
        glDeleteTextures(1,&stencilBuffer_);
    }
    stencilBuffer_ = 0;
    for (int i=0; i < NUM_STRATA; i++) {
        shaders_[i].clear();
        shaderStates_[i].clear();
    }
    gpu::vanilla::ConvLayerBase::cleanup();
}



/**
 * @brief Setup layer by allocating and initializing required GL resources
 *
 * @pre OpenGL context that is to be used for rendering must be current to the calling thread
 *
 * This function performs the required setup for layer operation. It allocates GL resources like
 * FBOs and VBOs, pre-computes the proxy-polygons used for rendering and also compiles all required
 * GL shaders to perform the computations on the proxy polygons. In addition, it creates the stencil
 * buffer (including contents upload) which is required to perform the convolution.
 *
 * @see setupStencilBuffers, setupShaders, setupFBOs, setupNetworkPolygons
 */
void TransConvLayerBase::setup() {
#ifdef DEBUG
    glGetError();
#endif
    setupStencilBuffer();
    setupShaders();
    setupFBOs();
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_, kernel_);
    vertexArray_->unbind();
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) {
        THROW_EXCEPTION_ARGS(FynException, "Failed to setup network layer (glerr=0x%x)", err);
    }
#endif
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void TransConvLayerBase::forward(uint64_t sequence) {
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (outputChanged_) updateFBOs();
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glDepthFunc(GL_ALWAYS);
    glDepthMask(GL_FALSE);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    glViewport(0, 0, viewport_[0], viewport_[1]);
    glStencilMask(0xFF);
    glClearColor(0,0,0,0);
    if (vertexArray_->bind()) {
        for (int outpass=0; outpass < weights_->numOutputRenderPasses(); outpass++) {
            framebuffers_.at(outpass)->bind();
            framebuffers_.at(outpass)->setWriteMask();
            setBias(outpass, weights_);
            performInputPasses(weights_, outpass);
            framebuffers_.at(outpass)->unbind();
        }
        vertexArray_->unbind();
    } else {
        FNLOGE("Cannot render layer %s",getName().c_str());
    }
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Apply configuration to unlinked shader program
 *
 * @param shader Shader program to be configured
 * @param stratum Rendering stratum
 *
 * @return Shared pointer to shader state which is to be applied to the shader before rendering
 *
 * @post Shader will be linked
 *
 * This function takes the (unlinked) shader program and:
 *   - binds attribute locations
 *   - creates a uniform state object that has the correct settings for the shader uniforms
 */
unistateptr TransConvLayerBase::configureShader(programptr shader,int stratum) const {
    shader->bindAttributeLocation("attributes0", 0);
    shader->bindAttributeLocation("attributes1", 1);
    shader->link();
    if (!shader->isLinked()) THROW_EXCEPTION_ARGS(FynException, "Could not link shader");
    shader->bind();
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        shader->mapUniformLocation("batchnorm", BATCHNORM_DATA);
    }
    if (outputPadding_ > 0) {
        shader->mapUniformLocation("bias", BIAS);
    }
    shader->mapUniformLocation("coeffs", COEFFICIENTS);
    unistateptr state = UniformState::makeShared(shader);

    if (stratum !=  0) {
        float hstep = 0.5f/(float)(width_ + 2 * inputPadding_);
        float vstep = 0.5f/(float)(height_ + 2 * inputPadding_);
        state->setUniformVec2("texStep", hstep, vstep);
    }
    shader->unbind();
    return state;
}


/**
 * @brief Execute render passes on the set of input textures
 *
 * @param weights Pointer to convolution weights
 * @param outputPass Output pass number (starts at 0 for first pass)
 *
 * This function renders the full batch of input channels for the provided output pass.
 */
void TransConvLayerBase::performInputPasses(UniformWeightArray *weights, int outputPass) {
    assert(upsample_ == 2);
    ShaderProgram *shader = nullptr;
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    int ibooffset=0;
    for (int stratum=0; stratum < NUM_STRATA; stratum++) {
        glStencilFuncSeparate(GL_FRONT_AND_BACK, GL_EQUAL, stratum+1, 0xFF);
        int xindex = stratum & 1;
        int yindex= (stratum & 2)>>1;
        ibooffset = stratum * 6 * sizeof(GLshort);
        shader = shaders_[stratum].at(weights->numRenderTargets(outputPass)).get();
        shader->bind(shaderStates_[stratum].at(weights->numRenderTargets(outputPass)).get());
        glActiveTexture(GL_TEXTURE0);
        for (int inpass=0; inpass < weights->numInputRenderPasses(); inpass++) {
            glBindTexture(GL_TEXTURE_2D,inputTextures_.at(inpass));
            const float *coeffs = weights->getPackageWeights(inpass,outputPass,xindex,yindex);
            shader->setMappedUniformMat4Array(COEFFICIENTS, coeffs, weights->numRenderTargets(outputPass));
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const char *)0+ibooffset);
        }
    }
    if (shader) shader->unbind();
}

/**
 * @brief Set bias value (for unpadded outputs)
 *
 * @param outPass Output rendering pass
 * @param bias Pointer to weight array that stored biases and weights
 *
 * Depending on whether or not output padding is selected, this preloads the target fraembuffers
 * with the bias values in case the output padding was zero. For non-zero paddings, the bias is
 * handled within the shader itself.
 */
void TransConvLayerBase::setBias(int outPass, const UniformWeightArray *bias) {
    if (outputPadding_ > 0) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    } else {
        const float *data = bias->getPackageBias(outPass);
        for (int i = 0; i < bias->numRenderTargets(outPass); i++) {
            glClearBufferfv(GL_COLOR, i, data + i * PIXEL_PACKING);
        }
    }
}


/**
 * @brief Convolution-specific shader preprocessing on source level
 *
 * @param[inout] preproc Pointer to target pre-processor string which will be used as preprocessor
 *                       definitions with GPULayerBase::compileShaderPair
 *
 * @param maxChars Maximum available characters in the \p preproc array
 *
 * @return Remaining capacity in \p preproc buffer
 *
 * This function constructs (parts of) a preprocessor string for use in the vertex and fragment
 * shaders. It currently takes care of the following things:
 *  - kernel size
 *  - shader-controller bias
 */
size_t TransConvLayerBase::shaderPreprocessing(char *preproc, size_t maxChars) {
    ssize_t mc = handlePreprocFlags(flags_, preproc, maxChars);
    if (outputPadding_ > 0) {
        strncat(preproc, "#define USE_BIAS\n", mc);
        mc -= maxChars - strlen(preproc);  // ouch
    }
    return mc;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void TransConvLayerBase::setupFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException, "No output texture set in convlayer %s", getName().c_str());
    if (!weights_) THROW_EXCEPTION_ARGS(FynException, "No weights loaded");
    int texoffset = 0;
    for (int pass = 0; pass < weights_->numOutputRenderPasses(); pass++) {
        FBO *fbo = new FBO(context_, viewport_[0], viewport_[1], outputTextures_.at(texoffset++));
        fbo->bind();
        for (int i = 1; i < weights_->numRenderTargets(pass); i++) {
            fbo->addTexture(GL_COLOR_ATTACHMENT0 + i, outputTextures_.at(texoffset++), GL_TEXTURE_2D);
        }
        fbo->addRenderbuffer(GL_DEPTH_STENCIL_ATTACHMENT, stencilBuffer_);
        fbo->setWriteMask();
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void TransConvLayerBase::updateFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException, "No output texture set in convlayer %s", getName().c_str());
    if (!weights_) THROW_EXCEPTION_ARGS(FynException, "No weights loaded");
    int texoffset = 0;
    for (int pass = 0; pass < weights_->numOutputRenderPasses(); pass++) {
        FBO *fbo = framebuffers_.at(pass);
        fbo->bind();
        for (int i = 0; i < weights_->numRenderTargets(pass); i++) {
            fbo->updateColorAttachment(GL_COLOR_ATTACHMENT0 + i, outputTextures_.at(texoffset++));
        }
        fbo->unbind();
    }
    outputChanged_ = false;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Set up proxy polygons for performing layer computation
 *
 * @param vao Pointer to VAO that stores the buffer state
 * @param kernel Kernel size for the convolution (currently ignored)
 *
 * @pre The vertex array object \p vao to be used with the created buffers is already bound
 *
 * This function sets up proxy polygons for rendering. Among coordinates for the target viewport,
 * these polygons also contain texture coordinates for the input data, as well as "depth" values
 * to be used for the stencil buffer that controls the broadcasting of the convolution weights
 * to the target texture.
 *
 * @see setupStencilBuffer
 */
void TransConvLayerBase::setupNetworkPolygons(VAO *vao, int kernel) {
    assert(upsample_ == 2);
    int vertsize = 3;
    int texsize = 2;
    float * attrs0 = new float[vertsize*NUM_STRATA*4];
    float * attrs1 = new float[texsize*NUM_STRATA*4];

    float posleft   = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[0]);
    float posright  =  1.0f - ((float)(2*outputPadding_) / (float)viewport_[0]);
    float postop    = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[1]);
    float posbottom =  1.0f - ((float)(2*outputPadding_) / (float)viewport_[1]);

    float thspan = (float)(width_) / (float)(width_ + 2*inputPadding_);
    float tvspan = (float)(height_) / (float)(height_ + 2*inputPadding_);
    float tleft  = ((float)inputPadding_) / (float)(width_ + 2*inputPadding_);
    float ttop   = ((float)inputPadding_) / (float)(height_ + 2*inputPadding_);

    int offset0 = 0;
    int offset1 = 0;
    for (int stratum=0; stratum < NUM_STRATA; stratum++) {
        float depth=0.0f;
        switch (stratum) {
        case 0:
            depth=0.125f+0.0625f;
            break;
        case 1:
            depth=0.250f+0.0625f;
            break;
        case 2:
            depth=0.375f+0.0625f;
            break;
        case 3:
            depth=0.500f+0.0625f;
            break;
        }
        attrs0[offset0 + 0*vertsize + 0] = posleft;
        attrs0[offset0 + 0*vertsize + 1] = postop;
        attrs0[offset0 + 0*vertsize + 2] = depth;
        attrs0[offset0 + 1*vertsize + 0] = posleft;
        attrs0[offset0 + 1*vertsize + 1] = posbottom;
        attrs0[offset0 + 1*vertsize + 2] = depth;
        attrs0[offset0 + 2*vertsize + 0] = posright;
        attrs0[offset0 + 2*vertsize + 1] = posbottom;
        attrs0[offset0 + 2*vertsize + 2] = depth;
        attrs0[offset0 + 3*vertsize + 0] = posright;
        attrs0[offset0 + 3*vertsize + 1] = postop;
        attrs0[offset0 + 3*vertsize + 2] = depth;
        offset0 += 4*vertsize;

        attrs1[offset1 + 0*texsize + 0] = tleft;
        attrs1[offset1 + 0*texsize + 1] = ttop;
        attrs1[offset1 + 1*texsize + 0] = tleft;
        attrs1[offset1 + 1*texsize + 1] = ttop+tvspan;
        attrs1[offset1 + 2*texsize + 0] = tleft+thspan;
        attrs1[offset1 + 2*texsize + 1] = ttop+tvspan;
        attrs1[offset1 + 3*texsize + 0] = tleft+thspan;
        attrs1[offset1 + 3*texsize + 1] = ttop;
        offset1 += 4*texsize;
    }
    coordBuffer_ = new VBO(context_);
    textureBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vao->enableArray(1);
    coordBuffer_->setBufferData(attrs0,vertsize*4*NUM_STRATA*sizeof(float),GL_STATIC_DRAW);
    coordBuffer_->bind();
    vao->setVertexAttributeBuffer(0,vertsize,GL_FLOAT,GL_FALSE,0,0);
    textureBuffer_->setBufferData(attrs1,texsize*4*NUM_STRATA*sizeof(float),GL_STATIC_DRAW);
    textureBuffer_->bind();
    vao->setVertexAttributeBuffer(1,texsize,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    delete [] attrs1;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[6*NUM_STRATA];
    indexBuffer_ = new IBO(context_);
    for (int i=0;i<NUM_STRATA;i++) {
        int offset=i*4;
        indices[i*6 + 0] = offset + 0;
        indices[i*6 + 1] = offset + 1;
        indices[i*6 + 2] = offset + 2;
        indices[i*6 + 3] = offset + 0;
        indices[i*6 + 4] = offset + 2;
        indices[i*6 + 5] = offset + 3;
    }
    indexBuffer_->setBufferData(indices, 6*NUM_STRATA*sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}



/**
 * @brief Setup stencil buffer to aid in broadcasting of convolution weights
 *
 * The transpose convolution is eseentially a scatter operation that adds the convolution kernel
 * to the target tensor in a regular fashion Currently ths transpose convolution layers in this
 * code only support stride-2 transpose-convolutions, wnich performs a "convoluted upsampling" of
 * the input tensor by a factor of 2 along both spatial dimensions. The fixed 2-fold upsampling
 * basically leads to 4 different configurations which are encoded in a stencil-buffer and 4
 * specialized shaders for each of the configurations.
 *
 * This function sets up that stencil buffer for a fixed upsampling factor of 2.
 */
void TransConvLayerBase::setupStencilBuffer() {
    using namespace opengl;
    assert(upsample_ == 2);
    glGenRenderbuffers(1, &stencilBuffer_);
    int err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(FynException,"Cannot setup stencil renderbuffer (err=0x%X)",err);
    //-----------------------------------------------
    // Setup FBO for rendering
    //-----------------------------------------------
    FBO *fbo = new FBO(context_, viewport_[0], viewport_[1], LayerBase::PIXEL_PACKING,Texture::UINT8);
    //-----------------------------------------------
    // Setup renderbuffer that will hold the stencil
    //-----------------------------------------------
    glBindRenderbuffer(GL_RENDERBUFFER,stencilBuffer_);
#ifndef ANDROID
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_STENCIL,viewport_[0],viewport_[1]);
#else
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,viewport_[0],viewport_[1]);
#endif
#ifdef DEBUG
    err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(FynException,"Cannot allocate depth/stencil renderbuffer (err=0x%X)",err);
#endif
    fbo->addRenderbuffer(GL_DEPTH_STENCIL_ATTACHMENT,stencilBuffer_);
    // NOTE (mw) not simply uploading a stencil texture here because that does not work on some hardware
    //-----------------------------------------------
    // Setup helper texture that will guide the depth
    // setup...
    //-----------------------------------------------
    GLuint helptex=0;
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1,&helptex);
    glBindTexture(GL_TEXTURE_2D,helptex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    bool odd = ((viewport_[0]&1)==1);
    if (odd) glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    unsigned char * helper = new unsigned char[viewport_[0]*viewport_[1]];
    memset(helper,0,viewport_[0]*viewport_[1]);
    for (int yi=0,y=outputPadding_; y < viewport_[1] - outputPadding_; y++,yi++) {
        for (int xi=0,x=outputPadding_; x < viewport_[0] - outputPadding_; x++,xi++) {
            helper[y*viewport_[0]+x] = 32*(1+((xi&1)+((yi&1)<<1)));
        }
    }
    glTexImage2D(GL_TEXTURE_2D,0,GL_R8,viewport_[0],viewport_[1],0,GL_RED,GL_UNSIGNED_BYTE,helper);
    if (odd) glPixelStorei(GL_UNPACK_ALIGNMENT,4);
    delete [] helper;
#ifdef DEBUG
    err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(FynException,"Error on texture and helper creation (0x%x)",err);
#endif
    //-----------------------------------------------
    // Setup shaders...
    //-----------------------------------------------
    static const char *vertshader = "precision mediump float;\n"
                                    "precision highp int;\n"
                                    "in vec4 attributes0;\n"
                                    "out vec2 texCoord;\n"
                                    "void main() {\n"
                                    "  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);\n"
                                    "  texCoord = vec2(attributes0.z,attributes0.w);\n"
                                    "}\n";
    static const char *fragshader = "precision mediump float;\n"
                                    "precision highp int;\n"
                                    "uniform mediump sampler2D helper;\n"
                                    "layout(location=0) out vec4 fragmentColor;\n"
                                    "uniform int pass;\n"
                                    "in vec2 texCoord;\n"
                                    "void main() {\n"
                                    "  float h = texture(helper,texCoord).r;\n"
                                    "  if (h==0.0) discard;\n"
                                    "  switch (pass) {\n"
                                    "    case 0:\n"
                                    "      if (h < 0.124) discard;\n"
                                    "      break;\n"
                                    "    case 1:\n"
                                    "      if (h < 0.24) discard;\n"
                                    "      break;\n"
                                    "    case 2:\n"
                                    "      if (h < 0.37) discard;\n"
                                    "      break;\n"
                                    "    case 3:\n"
                                    "      if (h < 0.49) discard;\n"
                                    "      break;\n"
                                    "  }\n"
                                    "  fragmentColor.rg=texCoord;\n"
                                    "  fragmentColor.b = h;\n"
                                    "  gl_FragDepth=h;\n"
                                    "}\n";
    shaderptr vs = shaderptr(new VertexShader(context_));
    shaderptr fs = shaderptr(new FragmentShader(context_));
    vs->setCode(vertshader);
    fs->setCode(fragshader);
    vs->compile();
    fs->compile();
    programptr shader = ShaderProgram::createInstance(context_);
    try {
        shader->addShader(vs);
        shader->addShader(fs);
        shader->link();
        shader->bind();
        shader->setUniformValue("helper",0);
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    //-----------------------------------------------
    // Setup VBO and polygon/texture coordinates...
    //-----------------------------------------------
    VAO *vao = new VAO(context_);
    vao->bind();
    VBO *vbo = new VBO(context_);
    float vbodata[16];
    vbodata[ 0+0]=-1.0f; // position (top left)
    vbodata[ 0+1]=-1.0f;
    vbodata[ 0+2]= 0.0f;
    vbodata[ 0+3]= 0.0f;
    vbodata[ 4+0]=-1.0f; // position (bottom left)
    vbodata[ 4+1]= 1.0f;
    vbodata[ 4+2]= 0.0f;
    vbodata[ 4+3]= 1.0f;
    vbodata[ 8+0]= 1.0f; // position (bottom right)
    vbodata[ 8+1]= 1.0f;
    vbodata[ 8+2]= 1.0f;
    vbodata[ 8+3]= 1.0f;
    vbodata[12+0]= 1.0f; // position (top right)
    vbodata[12+1]=-1.0f;
    vbodata[12+2]= 1.0f;
    vbodata[12+3]= 0.0f;
    vao->enableArray(0);
    vbo->setBufferData(vbodata,16*sizeof(float),GL_STATIC_DRAW);
    vbo->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    //-----------------------------------------------
    //-----------------------------------------------
    fbo->bind();
    glViewport(0,0,viewport_[0],viewport_[1]);
    glStencilFuncSeparate(GL_FRONT_AND_BACK,GL_ALWAYS,0,0xFF);
    glStencilMask(0xFF);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glDepthFunc(GL_ALWAYS);
    glStencilOp(GL_KEEP,GL_KEEP,GL_INCR);
    for (int pass=0;pass<4;pass++) {
        shader->setUniformValue("pass",pass);
        glDrawArrays(GL_TRIANGLE_FAN,0,4);
    }
    glDisable(GL_DEPTH_TEST);
    //-----------------------------------------------
    // ...and cleanup
    //-----------------------------------------------
    shader->unbind();
    fbo->unbind();
    vao->unbind();
    vbo->unbind();
    glDeleteTextures(1,&helptex);
    delete vbo;
    delete vao;
    delete fbo;
}


} // vanilla namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
