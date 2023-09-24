//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Linear Layer for Sequences (Header)                                         (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../common/miscdefs.h"
#include "linear_sequence.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
LinearLayer::LinearLayer(const LinearLayerBuilder &builder) : LinearLayer(builder, builder.number_) {
}


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
LinearLayer::LinearLayer(const LinearLayerBuilder &builder, int layerNumber)
        : GPULayerBase((GPULayerBuilder &) builder, layerNumber) {
    assert(builder.maxSequenceLen_ > 0);
    // For sequence processing, the height corresponds to the sequence length and the width to the
    // embedding dimension (divided by 4)
    dataType_ = builder.wgtType_;
    quantType_ = builder.quantType_;
    quantGroupSize_ = builder.quantGroupSize_;
    width_ = (inputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;    // input width
    height_ = builder.maxSequenceLen_;                              // input height
    viewport_[0] = width_;
    viewport_[1] = height_;
    hasBias_ = builder.hasBias_;
    matMul_ = new rudiments::MatMulConst(preprocessor_, inputChannels_, outputChannels_, height_,
                                         dataType_, quantGroupSize_,
                                         hasBias_, false, false, builder.context_);
    hasParameters_ = true;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void LinearLayer::cleanup() {
    FNET_DEL_AND_CLEAR(matMul_);
    GPULayerBase::cleanup();
}


/**
 * @copydoc LayerBase::setup
 */
void LinearLayer::setup() {
    matMul_->setup();
    setupFBOs();
    assert(glGetError() == GL_NO_ERROR);
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void LinearLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    if (!state) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() without token state");
    sequenceLength_ = state->seqLength;
    assert(glGetError() == GL_NO_ERROR);
    glEnable(GL_SCISSOR_TEST);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    matMul_->forward(sequenceLength_, 0, framebuffers_.at(0));
    glDisable(GL_SCISSOR_TEST);
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
std::vector<BufferSpec> LinearLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    int width = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    result.push_back(BufferSpec(0, 0,
                                width, height_,
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
std::vector<BufferSpec> LinearLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    // TODO (mw) handle residual inputs
    result.push_back(BufferSpec(0, 0,
                                width_, height_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    return result;
}

/**
 * @brief Load parameters from a parameter provider
 *
 * @param source Parameter provider to load parameters from
 *
 * This function loads parameters from the \p source, containing of weights, biases and also
 * quantization tables in case quantization is enabled. The parameters are accessed in the provider
 * using the following convention for the \c name and the \c subIndex:
 *   - \c layername.weights with a \c subIndex of 0 for the weights
 *   - \c layername.bias with a \c subIndex of 1 for the biases
 *   - \c layername.scales with a \c subIndex of 3 for the quantization scales
 *   - \c layername.zeros with a \c subIndex of 4 for the quantization zero-biases
 *
 * Where \c layername is the name that was given to the layer in the builder.
 */
void LinearLayer::loadParameters(const ParameterProvider * source) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    assert(matMul_);
    // TODO (mw) check the quantization mode
    matMul_->loadWeights(source->get(getName()+std::string(".weights"), getNumber(), 0));
    if (hasBias_) matMul_->loadBiases(source->get(getName()+std::string(".bias"), getNumber(), 0));
    if (quantType_ != qt_type::QT_NONE) {
        // FIXME (mw) check the quantization mode precisely
        auto scales = source->get(getName() + std::string(".scales"), getNumber(), 3);
        auto zeros = source->get(getName() + std::string(".zeros"), getNumber(), 4);
        matMul_->loadQuantizationTables(scales, zeros);
    }
}


/**
 * @copydoc LayerBase::writeResult
 */
void LinearLayer::writeResult(const char *fileName, bool includePadding) {
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


/**
 * @copydoc GPULayerBase::getGPUOutputBuffer
 */
GPUBuffer * LinearLayer::getGPUOutputBuffer(int port) const {
    if (outputTextures_.empty()) return nullptr;
    int width = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    auto * out = createGPUBuffer(width, height_, PIXEL_PACKING, getOutputOrder(port), getOutputType(port), 0);
    pushSliceToBuffer(out, outputTextures_[0], width, height_, PIXEL_PACKING, getOutputType(port));
    return out;
}


/**
 * @copydoc GPULayerBase::getGPUInputBuffer
 */
GPUBuffer * LinearLayer::getGPUInputBuffer(int port) const {
    if (inputTextures_.empty()) return nullptr;
    auto * out = createGPUBuffer(width_, height_, PIXEL_PACKING, getInputOrder(port), getInputType(port), 0);
    pushSliceToBuffer(out, inputTextures_[0], width_, height_, PIXEL_PACKING, getInputType(port));
    return out;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void LinearLayer::updateFBOs() {
    framebuffers_[0]->bind();
    framebuffers_[0]->updateColorAttachment(GL_COLOR_ATTACHMENT0, outputTextures_[0]);
    framebuffers_[0]->unbind();
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void LinearLayer::setupFBOs() {
    assert(outputTextures_.size() == 1);
    int width = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    framebuffers_.push_back(new FBO(context(), width, height_, outputTextures_.at(0)));
}

/**
 * @copydoc GPULayerBase::getInputOrder
 */
BufferSpec::order LinearLayer::getInputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order LinearLayer::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
