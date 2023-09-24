//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Embedding Layer for Sequences (Header)                                      (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vertexshader.h"
#include "../floatconversion.h"
#include "../../common/miscdefs.h"
#include "embedding_sequence.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerBuilder &builder) : EmbeddingLayer(builder, builder.number_) {
}


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerBuilder &builder, int layerNumber)
        : GPULayerBase((GPULayerBuilder &) builder, layerNumber) {
    assert(builder.maxSequenceLen_ > 0);
    embedDim_ = outputChannels_;
    width_ = 1;
    height_ = builder.maxSequenceLen_;
    tableRows_ = builder.tableRows_;
    assert(tableRows_ > 0);
    viewport_[0] = (embedDim_ + PIXEL_PACKING -1) / PIXEL_PACKING;
    viewport_[1] = height_;
    hasParameters_ = true;
}



/**
 * @copydoc GPULayerBase::cleanup
 */
void EmbeddingLayer::cleanup() {
    embeddingTextures_.clear();
    shader_.reset();
    FNET_DEL_AND_CLEAR(vertices_);
    FNET_DEL_AND_CLEAR(array_);
    GPULayerBase::cleanup();
}


/**
 * @copydoc GPULayerBase::setup
 */
void EmbeddingLayer::setup() {
    CLEAR_GFXERR_DEBUG
    proxyGeometry();
    setupFBOs();
    // NOTE (mw) this layer compiles its shader on-the-fly in the first call to forward()
    assert(glGetError() == GL_NO_ERROR);
    valid_ = true;
}



/**
 * @copydoc LayerBase::forward
 */
void EmbeddingLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    if (!state) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() without token state");
    if (!shader_) compileShader();
    sequenceLength_ = state->seqLength;
    CLEAR_GFXERR_DEBUG
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
    glEnable(GL_SCISSOR_TEST);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    for (size_t segment=0; segment < embeddingTextures_.size(); segment++) {
        glActiveTexture(GL_TEXTURE1+segment);
        glBindTexture(GL_TEXTURE_2D, embeddingTextures_[segment].getHandle());
    }
    array_->bind();
    glViewport(0, 0, viewport_[0], state->seqLength);
    glScissor(0, 0, viewport_[0], state->seqLength);
    framebuffers_.at(0)->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    shader_->bind();
    shader_->setUniformVec2("viewport", viewport_[0], state->seqLength);
    shader_->setUniformValue("textureHeight", embeddingTextures_[0].height());
    glDrawArrays(GL_LINES, 0, 2 * state->seqLength);
    framebuffers_.at(0)->unbind();
    shader_->unbind();
    array_->unbind();
    glDisable(GL_SCISSOR_TEST);
    disableTextureUnits((int)(embeddingTextures_.size() + 1));
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @brief Obtain buffer specifiers that are required as output for this layer
 *
 * @return Vector of buffer specifiers that specify the format for each required buffer
 *
 * @see BufferSpec
 *
 *
 * @note This layer differs from the standard 2D image layers found in FyuseNet. In particular, the
 *       viewport width stored in this layer is equivalent to the embedding size (divided by 4) and
 *       the viewport height is equivalent to the maximum sequence length.
 */
std::vector<BufferSpec> EmbeddingLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    return result;
}


/**
 * @brief Obtain buffer specifiers that are required as input for this layer
 *
 * @return Vector of buffer specifiers that specify the format for each required buffer
 *
 * @see BufferSpec
 *
 * @note This layer differs from the standard 2D image layers found in FyuseNet. In particular, the
 *       width stored in this layer is equivalent to the embedding size (divided by 4) and the height
 *       is equivalent to the maximum sequence length.
 */
std::vector<BufferSpec> EmbeddingLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                1, height_,
                                BufferSpec::sizedformat::SINGLE32UI, BufferSpec::genericformat::SINGLE, BufferSpec::dtype::UINT32,
                                BufferSpec::FUNCTION_SOURCE, 1).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    return result;
}


/**
 * @brief Load parameters from provider
 *
 * @param source ParameterProvider instance to load the data from
 *
 * This function retrieves the embedding vectors / vocabulary from the \p source and stores them in
 * a set of textures to be used in the lookup shader. The provider will be called with the following
 * parameters:
 *   - for \c name: \c layername.embed
 *   - for \c subIndex: 0
 */
void EmbeddingLayer::loadParameters(const ParameterProvider *source) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    using namespace opengl;
    assert(source);
    if (((embedDim_ + PIXEL_PACKING -1) / PIXEL_PACKING) > GLInfo::getMaximumTextureSize()) {
        THROW_EXCEPTION_ARGS(FynException,"Embedding dimension %d is too large for GPU", embedDim_);
    }
    DataBlob table = source->get(getName()+".embed", getNumber(), 0);
    assert(!table.empty());
    // -------------------------------------------------------
    // Select texture pixel format (currently no quantization
    // supported)...
    // -------------------------------------------------------
#ifdef HIGH_PRECISION
    opengl::Texture::pixtype pixtype = opengl::Texture::FLOAT32;
#else
    opengl::Texture::pixtype pixtype = opengl::Texture::FLOAT16;
#endif
    // -------------------------------------------------------
    // Allocate and fill textures...
    // -------------------------------------------------------
    int tgtsize = std::min((tableRows_  + HARD_TOKEN_TEXTURE_MAX-1) / HARD_TOKEN_TEXTURE_MAX,
                            GLInfo::getMaximumTextureSize());
    if (tgtsize & 1) tgtsize++;
    texWidth_ = (embedDim_ + PIXEL_PACKING -1) / PIXEL_PACKING;
    int numtextures = (tableRows_ + tgtsize - 1) / tgtsize;
    if (numtextures > HARD_TOKEN_TEXTURE_MAX) THROW_EXCEPTION_ARGS(FynException, "Vocabulary size (%d) too large", tableRows_);
    embeddingTextures_.reserve(numtextures+1);
    Texture::pixtype dtype = Texture::FLOAT32;
    int elsize = sizeof(float);
    const uint8_t * ptr = nullptr;
    switch (source->dataType(getName()+".embed", getNumber(), 0)) {
        case param_type::WGT_FLOAT32:
            ptr = reinterpret_cast<const uint8_t *>(std::any_cast<const uint16_t *>(table.get()));
            break;
        case param_type::WGT_FLOAT16:
            dtype = Texture::FLOAT16;
            elsize = sizeof(uint16_t);
            ptr = reinterpret_cast<const uint8_t *>(std::any_cast<const uint16_t *>(table.get()));
            break;
        case param_type::WGT_INT8:
            // intentional fallthrough
        case param_type::WGT_INT4:
            THROW_EXCEPTION_ARGS(FynException,"Data type not supported");
        default:
            break;
    }
    assert(ptr);
    // TODO (mw) handle the case when embedDim is not a multiple of 4
    for (int tex=0,rem=tableRows_; tex < numtextures; tex++, rem -= tgtsize) {
        int th = std::min(rem, tgtsize);
        opengl::Texture2D newtex(texWidth_, th, pixtype, 4);
        newtex.upload(ptr, dtype);
        ptr += th * embedDim_ * elsize;
        embeddingTextures_.push_back(newtex);
    }
    if (embeddingTextures_.empty()) THROW_EXCEPTION_ARGS(FynException, "Cannot create textures for embedding table (embed=%d height=%d)", embedDim_, tableRows_);
}



/**
 * @copydoc LayerBase::writeResult
 */
void EmbeddingLayer::writeResult(const char *fileName, bool includePadding) {
#ifdef DEBUG
    FBO * fbo = getFBO(0);
    int owidth = fbo->width();
    int oheight = fbo->height();
    int chans = PIXEL_PACKING;
#ifndef FYUSENET_USE_WEBGL
    FILE *out = fopen(fileName,"wb");
    if (out) {
        float * data = new float[owidth * oheight * chans];
#else
    uint8_t * download = new uint8_t[owidth * oheight * chans * sizeof(float)];
    float * data = (float *)download;
    if (true) {
#endif
        fbo->writeToMemory<float,GL_FLOAT>(data, chans, owidth * oheight * chans * sizeof(float));
#ifndef FYUSENET_USE_WEBGL
        fwrite(data, 1, owidth * sequenceLength_ * chans * sizeof(float), out);
        fclose(out);
        delete [] data;
#else
        EM_ASM({window.download($0, $1, $2);}, download, owidth * sequenceLength_ * chans * sizeof(float), fileName);
        delete [] download;
#endif
    }
#endif
}




/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shaders to perform operation on GPU
 */
void EmbeddingLayer::compileShader() {
    char preproc[128];
    using namespace opengl;
    CLEAR_GFXERR_DEBUG
    snprintf(preproc, sizeof(preproc), "#define VOCAB_SIZE %d\n", (int)embeddingTextures_.size());
    shader_ = ShaderRepository::compileShaderPair("shaders/sequence/seq_embed.vert", "shaders/sequence/seq_embed.frag", preproc, typeid(this), context());
    shader_->bindAttributeLocation("attributes0", 0);
    shader_->link();
    prepShader(shader_.get());
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @brief Perform some post-processing on the shader after compilation
 *
 * @param shader Shader to be adjusted
 *
 * Adjust the texture uniforms in case no binding support is present on the system
 */
void EmbeddingLayer::prepShader(ShaderProgram * shader) {
    assert(shader->isLinked());
    if (!opengl::GLInfo::hasBinding()) {
        shader->bind();
        shader->setUniformValue("inputTokens", 0);
        for (int i=0; i < (int)embeddingTextures_.size(); i++) {
            char name[32];
            snprintf(name, sizeof(name), "vocabulary%d", i);
            shader->setUniformValue(name, i+1);
        }
        shader->unbind();
    }
}


/**
 * @brief Create proxy geometry for the computation
 *
 * This creates a set of horizontal lines that are used to perform the "embedding lookup".
 */
void EmbeddingLayer::proxyGeometry() {
    using namespace opengl;
    array_ = new VAO(context());
    array_->bind();
    auto * attrs0 = new uint32_t[height_ * 2];
    for  (int row=0, offset=0; row < height_; row++) {
        attrs0[offset++] = (row << 1);
        attrs0[offset++] = (row << 1) | 1;
    }
    vertices_ = new VBO(context());
    array_->enableArray(0);
    vertices_->setBufferData(attrs0, (int)(height_ * 2 * sizeof(uint32_t)), GL_STATIC_DRAW);
    vertices_->bind();
    array_->setVertexAttributeBuffer(0, 1, GL_UNSIGNED_INT, 0, 0);
    delete [] attrs0;
    array_->unbind();
    vertices_->unbind();
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void EmbeddingLayer::updateFBOs() {
    framebuffers_[0]->bind();
    framebuffers_[0]->updateColorAttachment(GL_COLOR_ATTACHMENT0, outputTextures_[0]);
    framebuffers_[0]->unbind();
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void EmbeddingLayer::setupFBOs() {
    assert(outputTextures_.size() == 1);
    framebuffers_.push_back(new FBO(context(), viewport_[0], height_,
                                    outputTextures_.at(0)));
}


/**
 * @copydoc GPULayerBase::getInputOrder
 */
BufferSpec::order EmbeddingLayer::getInputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order EmbeddingLayer::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getInputType
 */
BufferSpec::dtype EmbeddingLayer::getInputType(int port) const {
    return BufferSpec::dtype::UINT32;
}


/**
 * @copydoc GPULayerBase::getOutputType
 */
BufferSpec::dtype EmbeddingLayer::getOutputType(int port) const {
    return GPULayerBase::TEXTURE_TYPE_DEFAULT;
}

} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
