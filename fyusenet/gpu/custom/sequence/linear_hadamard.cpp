//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Combination of a Linear Layer on top of Hadamard Product                    (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../../common/miscdefs.h"
#include "linear_hadamard.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::custom::sequence {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc GPULayerBase::GPULayerBase
 */
LinearHadamardLayer::LinearHadamardLayer(const CustomLayerBuilder &builder) : GPULayerBase((GPULayerBuilder &) builder) {
    using namespace gpu::sequence::rudiments;
    static auto shaderprep = [this](ShaderProgram * shader, MatMulConst::shtype type) {
        postprocShader(shader, type);
    };
    assert(builder.maxSequenceLen_ > 0);
    if (!builder.privData_.has_value()) THROW_EXCEPTION_ARGS(FynException, "No private data for layer %s (#%d)", builder.name_.c_str(), builder.number_);
    // For sequence processing, the height corresponds to the sequence length and the width to the
    // embedding dimension (divided by 4)
    auto priv = std::any_cast<const BuilderData &>(builder.privData_);
    dataType_ = priv.dataType_;
    quantType_ = priv.quantType_;
    hasBias_ = priv.hasBias_;
    quantGroupSize_ = priv.quantGroupSize_;
    width_ = (inputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;            // input width
    height_ = builder.maxSequenceLen_;                                      // input height
    viewport_[0] = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    viewport_[1] = height_;
    matMul_ = new MatMulConst(preprocessor_, inputChannels_, outputChannels_, height_, dataType_, quantGroupSize_,
                              hasBias_, (builder.getFlags() & LayerFlags::RESIDUAL_INPUT), false,
                              builder.context_);
    // FIXME (mw) we should to a runtime check here on the GLSL version instead
#if defined(HIGH_PRECISION) || defined(__APPLE__)
    matMul_->customShader(MatMulConst::shtype::FRAG_LONG, "shaders/custom/sequence/seq_hadamard_matmul_4bit_long.frag");
#else
    matMul_->customShader(MatMulConst::shtype::FRAG_LONG, "shaders/custom/sequence/seq_hadamard_matmul_4bit_long_half.frag");
#endif
    matMul_->customShader(MatMulConst::shtype::FRAG_SHORT, "shaders/custom/sequence/seq_hadamard_matmul_4bit_short.frag");
    matMul_->customShaderPostproc(shaderprep);
}



/**
 * @copydoc GPULayerBase::cleanup
 */
void LinearHadamardLayer::cleanup() {
    GLuint tex[4] = {weightData_, scaleData_, zeroData_, biasData_};
    glDeleteTextures(4, tex);
    FNET_DEL_AND_CLEAR(matMul_);
    GPULayerBase::cleanup();
}


/**
 * @copydoc GPULayerBase::setup
 */
void LinearHadamardLayer::setup() {
    CLEAR_GFXERR_DEBUG
    matMul_->setup();
    setupFBOs();
    assert(glGetError() == GL_NO_ERROR);
    valid_ = true;
}



/**
 * @copydoc LayerBase::forward
 */
void LinearHadamardLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    using mm = gpu::sequence::rudiments::MatMulConst;
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    if (!state) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() without token state");
    if (flags_ & LayerFlags::RESIDUAL_INPUT && residualTextures_.empty()) {
        THROW_EXCEPTION_ARGS(FynException, "No residual texture passed");
    }
    sequenceLength_ = state->seqLength;
    if (inputTextures_.size() != 2) THROW_EXCEPTION_ARGS(FynException, "Invalid number of input textures (need 2 found %d)", (int)inputTextures_.size());
    glEnable(GL_SCISSOR_TEST);
    for (int i=0; i < 2; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, inputTextures_.at(i));
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        glActiveTexture(GL_TEXTURE0 + mm::RESIDUAL_UNIT);
        glBindTexture(GL_TEXTURE_2D, residualTextures_.at(0));
    }
    matMul_->forward(state->seqLength, 0, framebuffers_.at(0));
    glDisable(GL_SCISSOR_TEST);
    disableTextureUnits(7);
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
 *       width stored in this layer is equivalent to the embedding size (divided by 4) and the height
 *       is equivalent to the maximum sequence length.
 */
std::vector<BufferSpec> LinearHadamardLayer::getRequiredOutputBuffers() const {
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
std::vector<BufferSpec> LinearHadamardLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    for (int i=0; i < 2; i++) {
        result.push_back(BufferSpec(0, i,
                                    width_, height_,
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        result.push_back(BufferSpec(0, 2,
                                    viewport_[0], viewport_[1],
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::RESIDUAL_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    }
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
 *  Where \c layername is the name that was assigned to this layer by the builder.
 */
void LinearHadamardLayer::loadParameters(const ParameterProvider * source) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    assert(matMul_);
    // TODO (mw) check the quantization mode
    matMul_->loadWeights(source->get(getName()+std::string(".weights"), getNumber(), 0));
    if (hasBias_) matMul_->loadBiases(source->get(getName()+std::string(".bias"), getNumber(), 1));
    if (quantType_ != qt_type::QT_NONE) {
        // TODO (mw) check the quantization mode precisely (we only support one right now)
        auto scales = source->get(getName() + std::string(".scales"), getNumber(), 3);
        auto zeros = source->get(getName() + std::string(".zeros"), getNumber(), 4);
        matMul_->loadQuantizationTables(scales, zeros);
    } else {
        THROW_EXCEPTION_ARGS(FynException, "Not supported yet");
    }
}


/**
 * @brief Generate custom builder for this layer
 *
 * @param name Name that should be assigned to this layer
 * @param bias Whether the operation should apply a bias to the linear part
 *
 * @return Pointer to builder which can be pushed to the layer factory
 */
CustomLayerBuilder * LinearHadamardLayer::createBuilder(const std::string& name, bool bias) {
    static auto geninstance = [](const CustomLayerBuilder& bld) {
        return (GPULayerBase * )(new LinearHadamardLayer(bld));
    };
    BuilderData priv;
    priv.hasBias_ = bias;
    auto * builder = new CustomLayerBuilder(name, geninstance);
    builder->privData_ = priv;
    return builder;
}

/**
 * @brief Generate custom builder for this layer
 *
 * @param name Name that should be assigned to this layer
 * @param quant Quantization type for the linear weights
 * @param dataType Data type for the linear weights
 * @param quantGroupSize Quantization group size for the linear weights
 * @param bias Whether the operation should apply a bias to the linear part
 *
 * @return Pointer to builder which can be pushed to the layer factory
 */
CustomLayerBuilder * LinearHadamardLayer::createBuilder(const std::string& name, qt_type quant, param_type dataType, int quantGroupSize , bool bias) {
    static auto geninstance = [](const CustomLayerBuilder& bld) {
        return (GPULayerBase * )(new LinearHadamardLayer(bld));
    };
    BuilderData priv;
    priv.hasBias_ = bias;
    priv.dataType_ = dataType;
    priv.quantType_ = quant;
    priv.quantGroupSize_ = quantGroupSize;
    auto * builder = new CustomLayerBuilder(name, geninstance);
    builder->privData_ = priv;
    return builder;
}


/**
 * @copydoc LayerBase::writeResult
 */
void LinearHadamardLayer::writeResult(const char *fileName, bool includePadding) {
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
GPUBuffer * LinearHadamardLayer::getGPUOutputBuffer(int port) const {
    if (outputTextures_.empty()) return nullptr;
    int width = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    auto * out = createGPUBuffer(width, height_, PIXEL_PACKING, getOutputOrder(port), getOutputType(port), 0);
    pushSliceToBuffer(out, outputTextures_[0], width, height_, PIXEL_PACKING, getOutputType(port));
    return out;
}


/**
 * @copydoc GPULayerBase::getGPUInputBuffer
 */
GPUBuffer * LinearHadamardLayer::getGPUInputBuffer(int port) const {
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
BufferSpec::order LinearHadamardLayer::getInputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order LinearHadamardLayer::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @brief Set shader parameters
 *
 * @param shader ShaderProgram to set parameters for
 * @param type shader type
 *
 * Sets texture uniforms on systems that do not support binding
 */
void LinearHadamardLayer::postprocShader(opengl::ShaderProgram * shader, gpu::sequence::rudiments::MatMulConst::shtype type) {
    assert(shader);
    if (!opengl::GLInfo::hasBinding()) {
        assert(shader->isBound());
        shader->setUniformValue("inputLayer0", 0);
        shader->setUniformValue("inputLayer1", 1);
        shader->setUniformValue("matrix", 2);
        shader->setUniformValue("scaleData", 3);
        shader->setUniformValue("zeroData", 4);
        shader->setUniformValue("biasData", 5, true);
        shader->setUniformValue("residual", 6, true);
    }
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void LinearHadamardLayer::updateFBOs() {
    framebuffers_[0]->bind();
    framebuffers_[0]->updateColorAttachment(GL_COLOR_ATTACHMENT0, outputTextures_[0]);
    framebuffers_[0]->unbind();
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void LinearHadamardLayer::setupFBOs() {
    assert(outputTextures_.size() == 1);
    int width = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    framebuffers_.push_back(new FBO(context(), width, height_, outputTextures_.at(0)));
}


} //  fyusion::fyusenet::gpu::custom::sequence namespace

// vim: set expandtab ts=4 sw=4:
