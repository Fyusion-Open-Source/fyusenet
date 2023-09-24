//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Transpose-Convolution Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/shaderprogram.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/uniformstate.h"
#include "../../gl/glinfo.h"
#include "../gfxcontextlink.h"
#include "../uniformweightarray.h"
#include "../floatconversion.h"

#include "deeptransconvlayerbase.h"
#include "deeplayerbase.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc DeepConvLayerBase::DeepConvLayerBase
 */
DeepTransConvLayerBase::DeepTransConvLayerBase(const ConvLayerBuilder& builder, int layerNumber):DeepConvLayerBase(builder, layerNumber) {
    assert(builder.upsample_[0] == builder.upsample_[1]);
    assert(builder.upsample_[0] == 2 && builder.upsample_[1] == 2);
    upsample_[0] = builder.upsample_[0];
    upsample_[1] = builder.upsample_[1];
    stencilBuffer_ = 0;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepTransConvLayerBase::cleanup() {
    if (stencilBuffer_) glDeleteRenderbuffers(1, &stencilBuffer_);
    DeepConvLayerBase::cleanup();
}


/**
 * @brief LayerBase::setup
 */
void DeepTransConvLayerBase::setup() {
    DeepConvLayerBase::setup();
    setupStencilBuffer();
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) {
        THROW_EXCEPTION_ARGS(FynException,"Failed to setup (deep) transconv layer (glerr=0x%x)",err);
    }
#endif
}


/**
 * @copydoc LayerBase::forward
 */
void DeepTransConvLayerBase::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glDepthFunc(GL_ALWAYS);
    glDepthMask(GL_FALSE);
    glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP);
    glBlendEquationSeparate(GL_FUNC_ADD,GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE);
    glStencilMask(0xFF);
    glClearColor(0,0,0,0);
    glViewport(0,0,viewport_[0],viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D,inputCoordTexture_);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D,weightTexture_);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);

    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        // TODO (mw) residual code here
    }

    for (int pass=0; pass < 4; pass++) {
        renderPass(pass);
    }

    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
}


/**
 * @copydoc DeepConvLayerBase::loadParameters()
 */
void DeepTransConvLayerBase::loadParameters(const ParameterProvider *weightSource) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    int texwidth = ((inputChannels_ % PIXEL_PACKING)==0) ? inputChannels_ : inputChannels_ + (PIXEL_PACKING - (inputChannels_ % PIXEL_PACKING));
    texwidth *= kernel_;
    if (texwidth & 1) texwidth++;
    int texheight = ((outputChannels_+(PIXEL_PACKING-1)) / PIXEL_PACKING)*kernel_;  // 4 pixels per matrix
    if (((texwidth/2) > GLInfo::getMaximumTextureSize()) || (texheight > GLInfo::getMaximumTextureSize())) {
        THROW_EXCEPTION_ARGS(FynException,"Weights do not fit into GL texture");
    }
    if (auto wgtsrc = weightSource->get(getName()+std::string(".weights"), getNumber(), 0) ; !wgtsrc.empty()) {
        float *weights = new float[texwidth * texheight * PIXEL_PACKING];
        memset(weights, 0, texwidth * texheight * PIXEL_PACKING * sizeof(float));
        const float *srcweights = std::any_cast<const float *>(wgtsrc.get());
        for (int outlayer = 0; outlayer < outputChannels_; outlayer += PIXEL_PACKING) {
            int orem = ((outputChannels_ - outlayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (outputChannels_ - outlayer);
            for (int fy = 0; fy < kernel_; fy++) {
                float *wptr = weights + ((outlayer / PIXEL_PACKING) * kernel_ + fy) * (texwidth * PIXEL_PACKING);
                // below defines one row in the target texture
                for (int inlayer = 0; inlayer < inputChannels_; inlayer += PIXEL_PACKING) {
                    int irem = ((inputChannels_ - inlayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_ - inlayer);
                    for (int fx = 0; fx < kernel_; fx++) {
                        for (int ol = outlayer; ol < outlayer + orem; ol++) {
                            for (int il = inlayer; il < inlayer + irem; il++) {
                                int srcoffset = ol * (kernel_ * kernel_ * inputChannels_) + ((fy * kernel_ + fx) * inputChannels_) + il;
                                *wptr = srcweights[srcoffset];
                                wptr++;
                            }
                            wptr += PIXEL_PACKING - irem;
                        }
                        wptr += (PIXEL_PACKING - orem) * PIXEL_PACKING;
                    }
                }
            }
        }
        if (!weightTexture_) glGenTextures(1, &weightTexture_);
        glBindTexture(GL_TEXTURE_2D, weightTexture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#ifndef HIGH_PRECISION
        if (halfSupport_) {
            unsigned int *fp16 = FloatConversion::getInstance()->toFP16UI(weights, texwidth * texheight * PIXEL_PACKING);
#ifdef GL_RGBA32UI
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, texwidth / 2, texheight, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, fp16);
#else
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32UI_EXT,texwidth/2,texheight,0,GL_RGBA_INTEGER_EXT,GL_UNSIGNED_INT,fp16);
#endif
            delete[] fp16;
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, texwidth, texheight, 0, GL_RGBA, GL_FLOAT, weights);
        }
#else
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,texwidth,texheight,0,GL_RGBA,GL_FLOAT,weights);
#endif
        delete[] weights;
    }
    //------------------------------------------------------
    // If we have the post-BN flag set, store the batchnorm
    // stuff...
    //------------------------------------------------------
    if (auto bnsrc = weightSource->get(getName()+std::string(".bn"), getNumber(), 2)  ; flags_ & fyusenet::LayerFlags::POST_BATCHNORM) {
        int padout = PIXEL_PACKING * ((outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING);
        const float * srcbn = std::any_cast<const float *>(bnsrc.get());
        postBNScales_ = new float[padout];
        postBNBias_ = new float[padout];
        memset(postBNScales_,0,padout*sizeof(float));
        memset(postBNBias_,0,padout*sizeof(float));
        memcpy(postBNScales_, srcbn, outputChannels_*sizeof(float));
        memcpy(postBNBias_, srcbn+outputChannels_, outputChannels_*sizeof(float));
    }
    //------------------------------------------------------
    // Now for the bias part (and also batchnorm)...
    //------------------------------------------------------
    int bs = PIXEL_PACKING * (1 + (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING);
    float * bias = new float[bs];
    memset(bias, 0, bs*sizeof(float));
    weightSource->map(getName()+std::string(".bias"), getNumber(), 1).with([&](const std::any &data) {
        memcpy(bias + PIXEL_PACKING, std::any_cast<const float *>(data), outputChannels_ * sizeof(float));
    });
    // load batchnorm scale and bias if necessary
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        for (int i=0;i<outputChannels_;i++) {
            bias[PIXEL_PACKING+i] = bias[PIXEL_PACKING+i] * postBNScales_[i] + postBNBias_[i];
            bias[PIXEL_PACKING+(bs/2)+i] = postBNScales_[i];
        }
    }
    if (!biasTexture_) glGenTextures(1,&biasTexture_);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#ifdef HIGH_PRECISION
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,1+(outputChannels_+PIXEL_PACKING-1)/PIXEL_PACKING,1,0,GL_RGBA,GL_FLOAT,bias);
#else
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,1 + (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING, 1, 0, GL_RGBA, GL_FLOAT, bias);
#endif
    delete [] bias;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepLayerBase::shaderPreprocessing
 */
size_t DeepTransConvLayerBase::shaderPreprocessing(char *preproc, size_t maxChars) {
#if defined(WIN32) || defined(WIN64)
            using ssize_t = int64_t;
#endif
    char extra[256];
    DeepConvLayerBase::shaderPreprocessing(preproc, maxChars);
    snprintf(extra, sizeof(extra), "#define DISP_UNIT %d\n#define WEIGHT_UNIT %d\n#define BIAS_UNIT %d\n",DISP_TEXTURE, WEIGHT_TEXTURE, BIAS_TEXTURE);
    strncat(preproc, extra, maxChars);
    return (size_t)std::max((ssize_t)0, (ssize_t)(maxChars-strlen(extra)));
}


/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Pointer to vertex-array-object that collects the buffers
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up the output tensor.
 *
 * @pre The vertex array object to be used with the network polygons is already bound
 */
void DeepTransConvLayerBase::setupNetworkPolygons(VAO *vao) {
    int offset0=0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    std::vector<DeepTiler::Tile> outtiles = tiler_->createOutputTiles();
    DeepTiler::Tile deftex = tiler_->getDefaultTextureExtents();
    //---------------------------------------------
    // VBO parts
    //---------------------------------------------
    for (DeepTiler::Tile tile : outtiles) {
        tile.toFloatVec(attrs0,offset0,4);
        deftex.toFloatVec(attrs0,offset0+2,4);
        offset0+=4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0,tiler_->numOutputTiles()*4*4*sizeof(float),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //
    int * attrs1 = new int[tiler_->numOutputTiles()*2*4];
    memset(attrs1,0,tiler_->numOutputTiles()*2*4*sizeof(float));
    for (int i=0;i<tiler_->numOutputTiles();i++) {
        for (int j=0;j<4;j++) {
            attrs1[(i*4+j)*2+0] = i;
            attrs1[(i*4+j)*2+1] = i;          // to be used for indexing bias texture
        }
    }
    textureOffsets_ = new VBO(context_);
    vao->enableArray(1);
    textureOffsets_->setBufferData(attrs1,tiler_->numOutputTiles()*2*4*sizeof(int),GL_STATIC_DRAW);
    textureOffsets_->bind();
    vao->setVertexAttributeBuffer(1,2,GL_INT,0,0);
    delete [] attrs1;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0;i<tiler_->numOutputTiles();i++) {
        int offset=i*4;
        indices[i*6+0]=offset+0;
        indices[i*6+1]=offset+1;
        indices[i*6+2]=offset+2;
        indices[i*6+3]=offset+0;
        indices[i*6+4]=offset+2;
        indices[i*6+5]=offset+3;
    }
    indexBuffer_->setBufferData(indices,6*tiler_->numOutputTiles()*sizeof(GLshort),GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
    //---------------------------------------------
    // Dependent texture to perform input lookup
    // in vertex shader...
    //---------------------------------------------
    glGenTextures(1,&inputCoordTexture_);
    glBindTexture(GL_TEXTURE_2D,inputCoordTexture_);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    float * texdata = new float[tiler_->numInputTiles()*4];
    DeepTiler::Tile defex = tiler_->getDefaultTextureExtents();
    std::vector<DeepTiler::Tile> tiles = tiler_->createInputTiles(0,0);
    int offset = 0;
    for (DeepTiler::Tile tile : tiles) {
        tile.toDisplacement(defex,texdata,offset);
        tile.lowClamp(texdata,offset+2);
        offset += 4;
    }
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,tiler_->numInputTiles(),1,0,GL_RGBA,GL_FLOAT,texdata);
    delete [] texdata;
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
void DeepTransConvLayerBase::setupStencilBuffer() {
    // NOTE (mw) only valid for stride 2
    glGenRenderbuffers(1, &stencilBuffer_);
    int err = glGetError();
    if (err != GL_NO_ERROR) {
        valid_ = false;
        THROW_EXCEPTION_ARGS(FynException, "Cannot setup stencil renderbuffer (err=0x%X)",err);
    }
    //-----------------------------------------------
    // Setup FBO for rendering
    //-----------------------------------------------
    FBO *fbo = new FBO(context_,viewport_[0],viewport_[1], 4, opengl::Texture::UINT8);
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
    bool odd = ((viewport_[0] & 1) == 1);
    if (odd) glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    unsigned char * helper = new unsigned char[viewport_[0]*viewport_[1]];
    memset(helper, 0, viewport_[0]*viewport_[1]);
    std::vector<DeepTiler::Tile> tiles = tiler_->createOutputTiles();
    for (DeepTiler::Tile tile : tiles) {
        for (int yi=0,y=tile.imageCoords_[1]; y < tile.imageCoords_[1]+tile.imageExtents_[1]; y++,yi++) {
            for (int xi=0,x=tile.imageCoords_[0]; x < tile.imageCoords_[0]+tile.imageExtents_[0]; x++,xi++) {
                helper[y*viewport_[0]+x] = 32*(1+((xi & 1) + ((yi & 1) <<1)));
            }
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, viewport_[0], viewport_[1], 0, GL_RED, GL_UNSIGNED_BYTE, helper);
    if (odd) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    delete [] helper;
#ifdef DEBUG
    err = glGetError();
    if (err != GL_NO_ERROR) {
        valid_ = false;
        THROW_EXCEPTION_ARGS(FynException, "Error on texture and helper creation (0x%x)",err);
    }
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
        valid_ = false;
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
    vbodata[ 0+0] = -1.0f; // position (top left)
    vbodata[ 0+1] = -1.0f;
    vbodata[ 0+2] =  0.0f;
    vbodata[ 0+3] =  0.0f;
    vbodata[ 4+0] = -1.0f; // position (bottom left)
    vbodata[ 4+1] =  1.0f;
    vbodata[ 4+2] =  0.0f;
    vbodata[ 4+3] =  1.0f;
    vbodata[ 8+0] =  1.0f; // position (bottom right)
    vbodata[ 8+1] =  1.0f;
    vbodata[ 8+2] =  1.0f;
    vbodata[ 8+3] =  1.0f;
    vbodata[12+0] =  1.0f; // position (top right)
    vbodata[12+1] = -1.0f;
    vbodata[12+2] =  1.0f;
    vbodata[12+3] =  0.0f;
    vao->enableArray(0);
    vbo->setBufferData(vbodata,16*sizeof(float),GL_STATIC_DRAW);
    vbo->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    //-----------------------------------------------
    //-----------------------------------------------
    fbo->bind();
    glViewport(0, 0, viewport_[0], viewport_[1]);
    glStencilFuncSeparate(GL_FRONT_AND_BACK,GL_ALWAYS,0,0xFF);
    glStencilMask(0xFF);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glDepthFunc(GL_ALWAYS);
    glStencilOp(GL_KEEP,GL_KEEP,GL_INCR);
    for (int pass=0; pass < 4; pass++) {
        shader->setUniformValue("pass",pass);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
    glDisable(GL_DEPTH_TEST);
    //-----------------------------------------------
    // ...and cleanup
    //-----------------------------------------------
    shader->unbind();
    fbo->unbind();
    vao->unbind();
    vbo->unbind();
    glDeleteTextures(1, &helptex);
    delete vbo;
    delete vao;
    delete fbo;
}

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
