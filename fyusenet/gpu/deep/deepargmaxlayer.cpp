//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep ArgMax Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cmath>
#include <cstring>
#include <cassert>
#include <cfloat>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../common/miscdefs.h"
#include "deepargmaxlayer.h"
#include "deeptiler.h"

namespace fyusion::fyusenet::gpu::deep {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
DeepArgMaxLayer::DeepArgMaxLayer(const ArgMaxLayerBuilder & builder, int layerNumber) :
    DeepLayerBase((const GPULayerBuilder &)builder, layerNumber) {
    assert(outputChannels_ <= 2);  // we allow up to two output channels here (first one is the index, 2nd one is the actual max value)
    if (builder.getFlags() & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException, "This layer does not support residual inputs");
#ifndef HIGH_PRECISION
    if (inputChannels_ > 2048) THROW_EXCEPTION_ARGS(FynException, "Due to the final output in 16-bit FP textures, this layer does not support more than 2048 input channels");
#endif
    channelBits_ = std::max(1, (int)ceil(log2((double)inputChannels_)));
    pass2Mask_ = ((1<<channelBits_)-1) << (EXPONENT_BITS + GUARD_BITS);
    pass1Mask_ = ~pass2Mask_;
}


/**
 * @copydoc LayerBase::cleanup
 */
void DeepArgMaxLayer::cleanup() {
    FNET_DEL_AND_CLEAR(pass1VBOA_);
    FNET_DEL_AND_CLEAR(pass1VBOB_);
    FNET_DEL_AND_CLEAR(pass1VBOC_);
    FNET_DEL_AND_CLEAR(pass1IBO_);
    FNET_DEL_AND_CLEAR(pass1VAO_);
    FNET_DEL_AND_CLEAR(pass2VAO_);
    FNET_DEL_AND_CLEAR(pass2VBO_);
    FNET_DEL_AND_CLEAR(pass2IBO_);
    FNET_DEL_AND_CLEAR(pass1FBO_);
    DeepLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::setup
 */
void DeepArgMaxLayer::setup() {
    setupNetworkPolygons();
    setupShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @brief Execute layer
 *
 * @param sequenceNo Sequence number (\b must be strictly increasing)
 * @param state Pointer to optional StateToken object that encapsulates per-run state information
 *
 * This function performs the actual computation that maps the input data to the output data
 * for this layer. The supplied \p sequenceNo number \b must be strictly increasing per network run
 * and may also be used for debugging purposes, in case errors only manifests themselves after a
 * certain number of computation cycles. It can also be used to keep track of the total number of
 * inference runs. Internally, it is used to make sure that asynchronously transmitted data is
 * up-to-date (on PBO reads for example).
 */
void DeepArgMaxLayer::forward(uint64_t sequenceNo, StateToken * state) {
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
    glBlendEquation(GL_MAX);
    float clear = (float)-powf(2,EXPONENT_MAX)-0.5f;
    glClearColor(clear, clear, clear, clear);
    glViewport(0, 0, viewport_[0], viewport_[1]);
    pass1FBO_->bind();
    pass1FBO_->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    pass1VAO_->bind();
    pass1Shader_->bind(pass1State_.get());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    glDrawElements(GL_TRIANGLES,tiler_->numInputTiles()*6,GL_UNSIGNED_SHORT,(const GLvoid *)nullptr);
    pass1Shader_->unbind(true);
    pass1VAO_->unbind();
    pass1FBO_->unbind();
    glDisable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    pass2VAO_->bind();
    pass2Shader_->bind(pass2State_.get());
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D,pass1FBO_->getAttachment());
    glDrawElements(GL_TRIANGLES,tiler_->numOutputTiles()*6,GL_UNSIGNED_SHORT,(const GLvoid *)nullptr);
    pass2VAO_->unbind();
    pass2Shader_->unbind();
}

/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepArgMaxLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, tiler_->getInputTextureWidth(),tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}

/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepArgMaxLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    // TODO (mw) in the future support writing out an RG texture here, however this will
    // require some adjustments in some layers too (like the download layer), as all layers
    // currently expect to be served with RGBA textures
    result.push_back(BufferSpec(0, 0, viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, BufferSpec::dtype::FLOAT32,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them.
 */
void DeepArgMaxLayer::setupShaders() {
#if defined(WIN32) || defined(WIN64)
        using ssize_t = int64_t;
#endif
    char preproc[1024] = {0}, line[512];
    ssize_t mc = (ssize_t)shaderPreprocessing(preproc, sizeof(preproc)-1);
    assert(mc > 0);
    float fmin = -FLT_MAX;
    // NOTE (mw) flt_min is a bit imprecise here, but we do not expect values that low
    snprintf(line,sizeof(line),"#define FLT_MIN %.8e\n#define PLACEMENT_BITS %d\n", fmin, EXPONENT_BITS+GUARD_BITS);
    strncat(preproc, line, mc);
    pass1Shader_ = compileShaderPair("shaders/deep/deepargmax.vert", "shaders/deep/deepargmax1.frag",preproc,typeid(this));
    pass1Shader_->bindAttributeLocation("attributes0", 0);
    pass1Shader_->bindAttributeLocation("attributes1", 1);
    pass1Shader_->bindAttributeLocation("attributes2", 2);
    pass1Shader_->link();
    pass1State_ = UniformState::makeShared(pass1Shader_);
    pass1State_->setUniformValue("inputLayer0", 0);
    pass1State_->setUniformVec4("bitmask", (GLint)pass1Mask_, (GLint)pass1Mask_, (GLint)pass1Mask_, (GLint)pass1Mask_);

    pass2Shader_ = compileShaderPair("shaders/deep/deepdefault.vert", "shaders/deep/deepargmax2.frag", preproc, typeid(this));
    pass2Shader_->bindAttributeLocation("attributes0",0);
    pass2Shader_->link();

    pass2State_ = UniformState::makeShared(pass2Shader_);
    pass2State_->setUniformValue("inputLayer0", 0);
    pass2State_->setUniformValue("bitmask", (GLint)pass2Mask_);
}


/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up the output tensor.
 */
void DeepArgMaxLayer::setupNetworkPolygons() {
    int offset0 = 0, offset1 = 0;
    float * attrs0 = new float[tiler_->numInputTiles()*4*4];
    unsigned int * attrs1 = new unsigned int[tiler_->numInputTiles()*4*4];
    unsigned int * attrs2 = new unsigned int[tiler_->numInputTiles()*4*4];
    pass1VAO_ = new VAO(context_);
    pass1VAO_->bind();
    //---------------------------------------------
    // VBO parts
    //---------------------------------------------
    std::vector<DeepTiler::Tile> otiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> itiles = tiler_->createInputTiles(0,0);
    for (int i=0; i < (int)itiles.size(); i++) {
        DeepTiler::Tile ot = otiles.at(0);
        DeepTiler::Tile it = itiles.at(i);
        ot.toFloatVec(attrs0,offset0,4);
        it.toFloatVec(attrs0,offset0+2,4);
        offset0 += 4*4;
        for (int j=0; j < 4; j++) {
            attrs2[offset1] = ((i*4) < inputChannels_) ? 1 : 0;
            attrs2[offset1+1] = ((i*4+1) < inputChannels_) ? 1 : 0;
            attrs2[offset1+2] = ((i*4+2) < inputChannels_) ? 1 : 0;
            attrs2[offset1+3] = ((i*4+3) < inputChannels_) ? 1 : 0;
            attrs1[offset1++] = i*4;
            attrs1[offset1++] = i*4+1;
            attrs1[offset1++] = i*4+2;
            attrs1[offset1++] = i*4+3;
        }
    }
    pass1VBOA_ = new VBO(context_);
    pass1VAO_->enableArray(0);
    pass1VBOA_->setBufferData(attrs0, (int)(tiler_->numInputTiles()*4*4*sizeof(float)),GL_STATIC_DRAW);
    pass1VBOA_->bind();
    pass1VAO_->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);

    pass1VBOB_ = new VBO(context_);
    pass1VAO_->enableArray(1);
    pass1VBOB_->setBufferData(attrs1, (int)(tiler_->numInputTiles()*4*4*sizeof(unsigned int)),GL_STATIC_DRAW);
    pass1VBOB_->bind();
    pass1VAO_->setVertexAttributeBuffer(1,4,GL_UNSIGNED_INT,0,0);

    pass1VBOC_ = new VBO(context_);
    pass1VAO_->enableArray(2);
    pass1VBOC_->setBufferData(attrs2, (int)(tiler_->numInputTiles()*4*4*sizeof(unsigned int)),GL_STATIC_DRAW);
    pass1VBOC_->bind();
    pass1VAO_->setVertexAttributeBuffer(2,4,GL_UNSIGNED_INT,0,0);

    delete [] attrs0;
    delete [] attrs1;
    delete [] attrs2;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numInputTiles()*6];
    pass1IBO_ = new IBO(context_);
    for (int i=0; i < tiler_->numInputTiles(); i++) {
        int offset = i*4;
        indices[i*6+0] = (GLshort)(offset+0);
        indices[i*6+1] = (GLshort)(offset+1);
        indices[i*6+2] = (GLshort)(offset+2);
        indices[i*6+3] = (GLshort)(offset+0);
        indices[i*6+4] = (GLshort)(offset+2);
        indices[i*6+5] = (GLshort)(offset+3);
    }
    pass1IBO_->setBufferData(indices, (int)(6 * tiler_->numInputTiles() * sizeof(GLshort)),GL_STATIC_DRAW);
    pass1IBO_->bind();
    delete [] indices;
    pass1VAO_->unbind();
    //---------------------------------------------
    // 2nd pass
    //---------------------------------------------
    pass2VAO_ = new VAO(context_);
    pass2VAO_->bind();
    DeepTiler::Tile unitext = DeepTiler::getUnitTextureExtents();
    assert(otiles.size() == 1);
    attrs0 = new float[otiles.size()*4*4];
    offset0 = 0;
    for (int i=0; i < (int)otiles.size(); i++) {
        DeepTiler::Tile ot = otiles.at(i);
        ot.toFloatVec(attrs0, offset0, 4);
        unitext.toFloatVec(attrs0, offset0+2, 4);
        offset0 += 4*4;
    }
    pass2VBO_ = new VBO(context_);
    pass2VAO_->enableArray(0);
    pass2VBO_->setBufferData(attrs0, (int)(tiler_->numOutputTiles() * 4 * 4 * sizeof(float)),GL_STATIC_DRAW);
    pass2VBO_->bind();
    pass2VAO_->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    indices = new GLshort[tiler_->numOutputTiles()*6];
    pass2IBO_ = new IBO(context_);
    for (int i=0; i < tiler_->numOutputTiles(); i++) {
        int offset = i*4;
        indices[i*6+0] = (GLshort)(offset+0);
        indices[i*6+1] = (GLshort)(offset+1);
        indices[i*6+2] = (GLshort)(offset+2);
        indices[i*6+3] = (GLshort)(offset+0);
        indices[i*6+4] = (GLshort)(offset+2);
        indices[i*6+5] = (GLshort)(offset+3);
    }
    pass2IBO_->setBufferData(indices, (int)(6 * tiler_->numOutputTiles() * sizeof(GLshort)),GL_STATIC_DRAW);
    pass2IBO_->bind();
    delete [] indices;
    pass2VAO_->unbind();
}


/**
 * @brief DeepArgMaxLayer::setupFBOs
 */
void DeepArgMaxLayer::setupFBOs() {
    DeepLayerBase::setupFBOs();
    // TODO (mw) check if system supports RG before using it ?
    pass1FBO_ = new FBO(context_,viewport_[0], viewport_[1], 2, opengl::Texture::FLOAT32);
}

} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
