//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Root Mean Square Norm for Sequences                                         (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>
#include <cstring>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/scoped_texturepool.h"
#include "../../gl/vertexshader.h"
#include "../floatconversion.h"
#include "../rudiments/proxygenerator.h"
#include "../../common/miscdefs.h"
#include "rmsnorm_sequence.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
RMSNormLayer::RMSNormLayer(const GPULayerBuilder &builder, int layerNumber)
        : GPULayerBase((GPULayerBuilder &) builder, layerNumber) {
    assert(builder.type_ == LayerType::RMSNORM);
    assert(builder.in() == builder.out());
    assert(inputChannels_ > 0);
    assert(builder.maxSequenceLen_ > 0);
    embedDim_ = inputChannels_;
    width_ = (embedDim_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    height_ = builder.maxSequenceLen_;
    const int maxcon = 16;          // TODO (mw) make this parameter dependent on the GPU type
    for (int con = maxcon; con >= 1; con--) {
        if (width_ % con == 0) {
            contraction_ = con;
            instances_ = width_ / contraction_;
            break;
        }
    }
    viewport_[0] = width_;
    viewport_[1] = height_;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void RMSNormLayer::cleanup() {
    FNET_DEL_AND_CLEAR(pass1ArrayLong_);
    FNET_DEL_AND_CLEAR(pass1VerticesLong_);
    FNET_DEL_AND_CLEAR(pass2ArrayLong_);
    FNET_DEL_AND_CLEAR(pass2VerticesLong_);
    FNET_DEL_AND_CLEAR(quadIndices_);
    FNET_DEL_AND_CLEAR(normFBO_);
    if (weightTexture_) glDeleteTextures(1, &weightTexture_);
    weightTexture_ = 0;
    pass1ShaderLong_.reset();
    pass2ShaderLong_.reset();
    shortShader_.reset();
    GPULayerBase::cleanup();
}


/**
 * @copydoc LayerBase::setup
 */
void RMSNormLayer::setup() {
    proxyGeometry();
    compileShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void RMSNormLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    if (!state) THROW_EXCEPTION_ARGS(FynException, "Sequence layers require state tokens");
    if (!weightTexture_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on layer without weights, run loadParameters() first");
    sequenceLength_ = state->seqLength;
    glEnable(GL_SCISSOR_TEST);
    if (sequenceLength_ <= 1) {
        computeShortSequence();
    } else {
        prepareRender();
        computeLongSequence();
    }
    glDisable(GL_SCISSOR_TEST);
}


/**
 * @brief Load weight data from a parameter provider
 *
 * @param source ParameterProvider instance to load the data from
 *
 * This function retrieves the weights for the RMS norm computation from the parameter provider. The
 * format of the data is expected to be one floating-point value per channel and will be accessed
 * by setting the \c name to \c layername.weights with a \c subIndex of 0.
 *
 *  @see ParameterProvider
 */
 void RMSNormLayer::loadParameters(const ParameterProvider *source) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    assert(source);
    CLEAR_GFXERR_DEBUG
    if (!weightTexture_) glGenTextures(1, &weightTexture_);
    if (!weightTexture_) THROW_EXCEPTION_ARGS(FynException, "Unable to create texture for weight texture (err 0x%x)", glGetError());
    glBindTexture(GL_TEXTURE_2D, weightTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    std::string myname = getName()+std::string(".weights");
    auto data = source->get(myname, getNumber(), 0);
    if (data.empty()) THROW_EXCEPTION_ARGS(FynException,"Empty data supplied to RMSNormLayer layer");
    GLenum buftype = (source->dataType(myname, getNumber(), 0) == param_type::WGT_FLOAT32) ? GL_FLOAT : GL_HALF_FLOAT;
    int esize = (buftype == GL_FLOAT) ? 4 : 2;
    const char * weights = (buftype == GL_FLOAT) ? reinterpret_cast<const char *>(std::any_cast<const float *>(data.get())) : reinterpret_cast<const char *>(std::any_cast<const uint16_t *>(data.get()));
    assert(weights);
    bool copy = false;
    if ((embedDim_ % PIXEL_PACKING != 0)) {
        char * buf = new char[width_ * PIXEL_PACKING * esize];
        memcpy(buf, weights, embedDim_ * esize);
        for (int i=embedDim_ * esize; i < width_ * PIXEL_PACKING * esize; i++) buf[i] = 0.f;
        weights = buf;
        copy = true;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, (buftype == GL_FLOAT) ? GL_RGBA32F : GL_RGBA16F, width_, 1, 0, GL_RGBA, buftype, weights);
    assert(glGetError() == GL_NO_ERROR);
    if (copy) delete [] weights;
}


/**
 * @brief Obtain buffer specifiers that are required as output for this layer
 *
 * @return Vector of buffer specifiers that specify the format for each required buffer
 *
 * @see BufferSpec
 *
 * @note This layer differs from the standard 2D image layers found in FyuseNet. In particular, the
 *       width stored in this layer is equivalent to the embedding size (divided by 4) and the height
 *       is equivalent to the maximum sequence length.
 */
std::vector<BufferSpec> RMSNormLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                width_, height_,
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
std::vector<BufferSpec> RMSNormLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                width_, height_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    return result;
}


/**
 * @copydoc LayerBase::writeResult
 */
void RMSNormLayer::writeResult(const char *fileName, bool includePadding) {
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
        fbo->writeToMemory<float,GL_FLOAT>(data, chans, (GLsizei)(owidth * oheight * chans * sizeof(float)));
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


/**
 * @copydoc GPULayerBase::getGPUOutputBuffer
 */
GPUBuffer * RMSNormLayer::getGPUOutputBuffer(int port) const {
    if (outputTextures_.empty()) return nullptr;
    auto * out = createGPUBuffer(width_, height_, PIXEL_PACKING, getOutputOrder(port), getOutputType(port), 0);
    pushSliceToBuffer(out, outputTextures_[0], width_, height_, PIXEL_PACKING, getOutputType(port));
    return out;
}


/**
 * @copydoc GPULayerBase::getGPUInputBuffer
 */
GPUBuffer * RMSNormLayer::getGPUInputBuffer(int port) const {
    if (inputTextures_.empty()) return nullptr;
    auto * out = createGPUBuffer(width_, height_, PIXEL_PACKING, getInputOrder(port), getInputType(port), 0);
    pushSliceToBuffer(out, inputTextures_[0], width_, height_, PIXEL_PACKING, getInputType(port));
    return out;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::getInputOrder
 */
BufferSpec::order RMSNormLayer::getInputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order RMSNormLayer::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @brief Compute RMS norm for single token (single matrix row)
 *
 * This executes a single render pass where the shader computes the norm (denominator) for the
 * normalization inside the vertex shader which is then passed to the fragment shader that
 * performs the actual weighting/normalization.
 *
 * A drawback of this approach is that the vertex shader will not scale to SMs, but this should
 * be negligible for the small amount of data we are processing here.
 */
void RMSNormLayer::computeShortSequence() {
    assert(embedDim_ > 0);
    assert(sequenceLength_ == 1);
    glDisable(GL_BLEND);
    glViewport(0, 0, width_, sequenceLength_);
    glScissor(0, 0, width_, sequenceLength_);
    pass1ArrayLong_->bind();
    framebuffers_.at(0)->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, weightTexture_);
    shortShader_->bind();
    shortShader_->setUniformVec2("embedWidth", width_, embedDim_);
    shortShader_->setUniformValue("row", (int)0);
    glDrawArrays(GL_LINES, 0, 2);
    pass1ArrayLong_->unbind();
    shortShader_->unbind();
    framebuffers_.at(0)->unbind();
}



/**
 * @brief Compute RMS norm for multiple tokens
 *
 * This executes two shader passes for the norm computation. The first pass computes the norm
 * (denominator) for each row using instanced rendering. The second pass uses the previously
 * computed norms - stored in a texture - and applies them to each row together with the
 * weighting.
 */
void RMSNormLayer::computeLongSequence() {
    assert(embedDim_ > 0);
    glLineWidth(1.0f);
    // --------------------------------------------------------
    // Pass 1: compute normalizer
    // --------------------------------------------------------
    glViewport(0, 0, sequenceLength_, 1);
    glScissor(0, 0, sequenceLength_, 1);
    pass1ArrayLong_->bind();
    pass1ShaderLong_->bind();
    pass1ShaderLong_->setUniformValue("contraction", contraction_);
    pass1ShaderLong_->setUniformVec2("inputSize", (float)width_, (float)sequenceLength_);
    normFBO_->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    glDrawArraysInstanced(GL_LINES, 0, 2, instances_);
    normFBO_->unbind();
    pass1ShaderLong_->unbind(true);
    pass1ArrayLong_->unbind();
    // --------------------------------------------------------
    // Pass 2: normalize
    // --------------------------------------------------------
    glDisable(GL_BLEND);
    glViewport(0, 0, width_, sequenceLength_);
    glScissor(0, 0, width_, sequenceLength_);
    framebuffers_.at(0)->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    pass2ArrayLong_->bind();
    pass2ShaderLong_->bind();
    pass2ShaderLong_->setUniformVec2("viewport", (float)width_, (float)sequenceLength_);
    pass2ShaderLong_->setUniformValue("scale", 1.0f / (float)embedDim_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normFBO_->getAttachment(GL_COLOR_ATTACHMENT0));
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, weightTexture_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr);
    pass2ShaderLong_->unbind();
    framebuffers_.at(0)->unbind();
    pass2ArrayLong_->unbind();
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void RMSNormLayer::updateFBOs() {
    framebuffers_[0]->bind();
    framebuffers_[0]->updateColorAttachment(GL_COLOR_ATTACHMENT0, outputTextures_[0]);
    framebuffers_[0]->unbind();
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void RMSNormLayer::setupFBOs() {
    uint32_t scope = (context().texturePool()) ? context().texturePool()->scopeID() : 0;
    normTex_ = Texture2D(embedDim_, 1, opengl::Texture::FLOAT32, 1, context().texturePool(), scope, false);
    normFBO_ = new FBO(context(), normTex_);
    assert(outputTextures_.size() == 1);
    framebuffers_.push_back(new FBO(context(), width_, height_, outputTextures_.at(0)));
}


/**
 * @brief Generate proxy geometry for computing RMS norm(s)
 *
 * Generates a simple line that is used as instance template for the 1st pass targeted at long
 * sequences (also used for the single pass for 1-token sequences) and a simple quad for the 2nd
 * pass for long sequences.
 */
void RMSNormLayer::proxyGeometry() {
    pass1ArrayLong_ = new VAO(context());
    pass1ArrayLong_->bind();
    // ------------------------------------------------------
    // Pass 1 proxy geometry data, for multiple tokens, we
    // use a simple line as instance template and define
    // it via the texture coordinates. This is also used for
    // the single-token version..
    // ------------------------------------------------------
    float p1lverts[] = {0.f, 0.f, 0.f, 1.f};
    pass1VerticesLong_ = new VBO(context());
    pass1ArrayLong_->enableArray(0);
    pass1VerticesLong_->setBufferData(p1lverts, sizeof(p1lverts), GL_STATIC_DRAW);
    pass1VerticesLong_->bind();
    pass1ArrayLong_->setVertexAttributeBuffer(0, 2, GL_FLOAT, false, 0, 0);
    pass1ArrayLong_->unbind();
    // ------------------------------------------------------
    // Pass 2 proxy geometry, this renders a simple quad
    // ------------------------------------------------------
    const auto [p2arr, p2verts, p2inds] = rudiments::ProxyGenerator::simpleQuad(context());
    pass2ArrayLong_ = p2arr;
    pass2VerticesLong_ = p2verts;
    quadIndices_ = p2inds;
}



/**
 * @brief Compile shaders for RMSNormLayer computation
 *
 * Compiles a total of 3 shaders (1 for short sequences, 2 for long sequences).
 */
void RMSNormLayer::compileShaders() {
    char preproc[256] = {0};
    auto texbind = [](ShaderProgram * shader) {
        shader->bind();
        shader->setUniformValue("inputLayer0", 0);
        shader->unbind();
    };
    pass1ShaderLong_ = compileShaderPair("shaders/sequence/rmsnorm_long_pass1.vert", "shaders/sequence/rmsnorm_long_pass1.frag", preproc, typeid(this));
    pass1ShaderLong_->bindAttributeLocation("attributes0", 0);
    pass1ShaderLong_->link();
    assert(pass1ShaderLong_->isLinked());
    pass2ShaderLong_ = compileShaderPair("shaders/sequence/rmsnorm_long_pass2.vert", "shaders/sequence/rmsnorm_long_pass2.frag", preproc, typeid(this));
    pass2ShaderLong_->bindAttributeLocation("attributes0", 0);
    pass2ShaderLong_->link();
    assert(pass2ShaderLong_->isLinked());
    shortShader_ = compileShaderPair("shaders/sequence/rmsnorm_short.vert", "shaders/sequence/rmsnorm_short.frag", preproc, typeid(this));
    shortShader_->bindAttributeLocation("attributes0", 0);
    shortShader_->link();
    assert(shortShader_->isLinked());
    if (!GLInfo::hasBinding()) {
        texbind(pass1ShaderLong_.get());
        texbind(shortShader_.get());
        pass2ShaderLong_->bind();
        pass2ShaderLong_->setUniformValue("inputLayer0", 0);
        pass2ShaderLong_->setUniformValue("normData", 1);
        pass2ShaderLong_->setUniformValue("weights", 2);
        pass2ShaderLong_->unbind();
        shortShader_->bind();
        shortShader_->setUniformValue("weights", 1);
        shortShader_->unbind();
    }
}



} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
