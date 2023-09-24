//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Multi-Head Causally-Masked Self-Attention Layer                             (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/scoped_texturepool.h"
#include "../floatconversion.h"
#include "../../common/miscdefs.h"
#include "../rudiments/proxygenerator.h"
#include "causal_multihead_attentionlayer.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
CausalMultiHeadAttentionLayer::CausalMultiHeadAttentionLayer(const AttentionLayerBuilder &builder, int layerNumber)
        : GPULayerBase((GPULayerBuilder &) builder, layerNumber) {
    using namespace gpu::rudiments;
    assert(builder.headDim_ > 0);
    assert((builder.headDim_ % PIXEL_PACKING) == 0);
    assert(builder.numHeads_ > 0);
    assert((builder.numHeads_ % PIXEL_PACKING) == 0);
    assert(builder.in() > 0);
    assert(builder.out() > 0);
    assert(builder.maxSequenceLen_ > 0);
    assert(builder.inputPadding_ == 0);
    assert(builder.outputPadding_ == 0);
    // ------------------------------------------------
    // Copy data from the builder...
    // ------------------------------------------------
    width_ = (inputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    height_ = builder.maxSequenceLen_;
    headDim_ = builder.headDim_;
    numHeads_ = builder.numHeads_;
    headDim_ = builder.headDim_;
    embedDim_ = builder.in();
    quantType_ = builder.quantType_;
    dataType_ = builder.wgtType_;
    quantGroupSize_ = builder.quantGroupSize_;
    posEnc_ = builder.posEncoding_;
    incremental_ = builder.incremental_;
    maxSequenceLength_ = builder.maxSequenceLen_;
    autoResidual_ = builder.autoResidual_;
    viewport_[0] = width_;
    viewport_[1] = height_;
    // ------------------------------------------------
    // Setup rudiments...
    // ------------------------------------------------
    bool inres = (builder.getFlags() & LayerFlags::RESIDUAL_INPUT);
    rotaryEncoder_ = new rudiments::RotaryEncoder(width_, headDim_, builder.thetaBase_, builder.context_);
    softMaxBatched_ = new rudiments::MaskedSoftMaxBatched(height_, MAX_DP_BATCH, builder.context_);
    softMaxSingle_ = new rudiments::MaskedSoftMaxSingle(numHeads_, headDim_, builder.context_);
    attMulBatched_ = new rudiments::AttentionMulBatched(numHeads_, headDim_, builder.maxSequenceLen_, builder.context_);
    attMulSingle_ = new rudiments::AttentionMulSingle(width_, numHeads_, headDim_, builder.context_);
    dotProdBatched_ = new rudiments::DotProductBatched(numHeads_, headDim_, MAX_DP_BATCH, builder.context_);
    dotProdSingle_ = new rudiments::DotProductSingle(width_, numHeads_, headDim_, builder.context_);
    queryMul_ = new rudiments::MatMulConst(PreambleGenerator(), embedDim_, numHeads_ * headDim_, maxSequenceLength_, dataType_, quantGroupSize_, false, false, false, builder.context_);
    keyMul_ = new rudiments::MatMulConst(PreambleGenerator(), embedDim_, numHeads_ * headDim_, maxSequenceLength_, dataType_, quantGroupSize_, false, false, false, builder.context_);
    valueMul_ = new rudiments::MatMulConst(PreambleGenerator(), embedDim_, numHeads_ * headDim_, maxSequenceLength_, dataType_, quantGroupSize_, false, false, false, builder.context_);
    outMul_ = new rudiments::MatMulConst(PreambleGenerator(), numHeads_ * headDim_, embedDim_, maxSequenceLength_, dataType_, quantGroupSize_, false, inres, builder.autoResidual_, builder.context_);
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void CausalMultiHeadAttentionLayer::cleanup() {
    // ------------------------------------------------
    // Clear rudiments...
    // ------------------------------------------------
    FNET_DEL_AND_CLEAR(rotaryEncoder_);
    FNET_DEL_AND_CLEAR(softMaxBatched_);
    FNET_DEL_AND_CLEAR(softMaxSingle_);
    FNET_DEL_AND_CLEAR(attMulBatched_);
    FNET_DEL_AND_CLEAR(attMulSingle_);
    FNET_DEL_AND_CLEAR(dotProdBatched_);
    FNET_DEL_AND_CLEAR(dotProdSingle_);
    FNET_DEL_AND_CLEAR(queryMul_);
    FNET_DEL_AND_CLEAR(keyMul_);
    FNET_DEL_AND_CLEAR(valueMul_);
    FNET_DEL_AND_CLEAR(outMul_);
    // ------------------------------------------------
    // Clear FBOs...
    // ------------------------------------------------
    FNET_DEL_AND_CLEAR(attValFBO_);
    FNET_DEL_AND_CLEAR(peQueryFBO_);
    FNET_DEL_AND_CLEAR(peKeyFBO_);
    FNET_DEL_AND_CLEAR(dotProdFBO_);
    FNET_DEL_AND_CLEAR(smPass2BatchFBO_);
    for (auto *fbo: qkvFBOs_) delete fbo;
    qkvFBOs_.clear();
    GPULayerBase::cleanup();
}



/**
 * @copydoc GPULayerBase::setup
 */
void CausalMultiHeadAttentionLayer::setup() {
    if (rotaryEncoder_) rotaryEncoder_->setup();
    attMulBatched_->setup();
    attMulSingle_->setup();
    dotProdBatched_->setup();
    dotProdSingle_->setup();
    queryMul_->setup();
    keyMul_->setup();
    valueMul_->setup();
    outMul_->setup();
    // NOTE (mw) some auxiliary functions are set up in setupFBOs()
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void CausalMultiHeadAttentionLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!state) THROW_EXCEPTION_ARGS(FynException, "Sequence layers require state tokens");
    if (state->seqLength <= 0) THROW_EXCEPTION_ARGS(FynException, "Illegal sequenceNo length %d supplied", state->seqLength);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    if (state->seqLength > height_) THROW_EXCEPTION_ARGS(FynException, "Query too long (%d), max is %d", state->seqLength, height_);
    if (incremental_ && (!state->reset) && (state->seqLength + keyLength_ > height_)) THROW_EXCEPTION_ARGS(FynException, "Incremental query too long (%d), max is %d (cached: %d)", state->seqLength, height_ - keyLength_, keyLength_);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        if (residualTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"Need residual input");
    }
    queryLength_ = (int)state->seqLength;
    tokenIndex_ = state->seqIndex;
    prepareRender();
    glEnable(GL_SCISSOR_TEST);
    compute();
    glDisable(GL_SCISSOR_TEST);
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
std::vector<BufferSpec> CausalMultiHeadAttentionLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                width_, height_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_SEQUENCE).passThrough(true));
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
std::vector<BufferSpec> CausalMultiHeadAttentionLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                width_, height_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        result.push_back(BufferSpec(0, 1,
                                    width_, height_,
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::RESIDUAL_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    }
    return result;
}


/**
 * @brief Load attention matrix parameters and quantization data for this layer
 *
 * @param weights Parameter data provider
 *
 * This function parses the weights, biases and quantization data stored in the \p weights parameter
 * for usage with the GPU. Most importantly, the storage order of the supplied weights is supposed
 * to be \b row-major (i.e. the first \f$ m \f$ elements are the first row of \f$ \mathbf{W} \f$
 * and so forth).
 *
 * On \e quantized weights, we assume that quantization is done by packing data into 32-bit
 * words in an LSB-first fashion. To provide an example, when using 8-bit quantization we may
 * consider the 32-bit word as an array of 4 values. The byte that translates to array index
 * 0 would then be the \e lowest byte (also known as little-endian order), i.e. assuming an
 * array of the form:
 * @code
 * uint8_t array[4] = {0, 1, 2, 3}
 * @endcode
 *
 * This would be stored as \c 0x03020100 inside a single 32-bit word. For unknown reasons
 * it is common practice for this quantized type of storage to have each 32-bit word represent
 * a <i>partial column</i>, i.e. the first \e n rows of a column and not the first \e n elements
 * of a column.
 *
 * @note It is safe to call this function from a context that is shared with the initial GL
 *       context that was used to create the layer.
 *
 * As this class requires quite a bit of parameters because of the compounding, the \p weights
 * is accessed with the following values for \c name and \c subIndex:
 *   - \c layername.query.weights (\c subIndex = 0) for the query matrix weights
 *   - \c layername.query.bias (\c subIndex = 1) for the query matrix biases
 *   - \c layername.query.scales (\c subIndex = 2) for the query matrix quantization scales
 *   - \c layername.query.zeros (\c subIndex = 3) for the quantized query matrix quantization zero-biases
 *   - \c layername.key.weights (\c subIndex = 4) for the key matrix weights
 *   - \c layername.key.bias (\c subIndex = 5) for the key matrix biases
 *   - \c layername.key.scales (\c subIndex = 6) for the key matrix quantization scales
 *   - \c layername.key.zeros (\c subIndex = 7) for the quantized key matrix quantization zero-biases
 *   - \c layername.value.weights (\c subIndex = 8) for the value matrix weights
 *   - \c layername.value.bias (\c subIndex = 9) for the value matrix biases
 *   - \c layername.value.scales (\c subIndex = 10) for the value matrix quantization scales
 *   - \c layername.value.zeros (\c subIndex = 11) for the quantized value matrix quantization zero-biases
 *   - \c layername.out.weights (\c subIndex = 12) for the output matrix weights
 *   - \c layername.out.bias (\c subIndex = 13) for the output matrix biases
 *   - \c layername.out.scales (\c subIndex = 14) for the output matrix quantization scales
 *   - \c layername.out.zeros (\c subIndex = 15) for the quantized output matrix quantization zero-biases
 *
 * Where \c layername is the name that was assigned to this layer by the builder.
 *
 * @warning See storage order assumption in the long description
 */
void CausalMultiHeadAttentionLayer::loadParameters(const ParameterProvider * weights) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    std::string suffix[4] = {".query", ".key", ".value", ".out"};
    rudiments::MatMulConst * mul[4] = {queryMul_, keyMul_, valueMul_, outMul_};
    for (int sub = QUERY; sub <= OUTPUT; sub++) {
        int sidx = sub * 4;
        DataBlob wgtblob = weights->get(getName()+suffix[sub]+std::string(".weights"), getNumber(), sidx);
        mul[sub]->loadWeights(wgtblob);
        if (hasBias_) {
            DataBlob bsblob = weights->get(getName()+suffix[sub]+std::string(".bias"), getNumber(), sidx+1);
            mul[sub]->loadBiases(bsblob);
        }
        if (dataType_ != param_type::WGT_FLOAT) {
            auto scales = weights->get(getName()+suffix[sub]+std::string(".scales"), getNumber(), sidx+2);
            auto zeros = weights->get(getName()+suffix[sub]+std::string(".zeros"), getNumber(), sidx+3);
            assert(!scales.empty());
            assert(!zeros.empty());
            mul[sub]->loadQuantizationTables(scales, zeros);
        }
    }
}


/**
 * @copydoc LayerBase::writeResult
 */
void CausalMultiHeadAttentionLayer::writeResult(const char *fileName, bool includePadding) {
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
        fwrite(data, 1, owidth * queryLength_ * chans * sizeof(float), out);
        fclose(out);
        delete [] data;
#else
        EM_ASM({window.download($0, $1, $2);}, download, owidth * queryLength_ * chans * sizeof(float), fileName);
        delete [] download;
#endif
    }
#endif
}


/**
 * @copydoc GPULayerBase::getGPUOutputBuffer
 */
GPUBuffer * CausalMultiHeadAttentionLayer::getGPUOutputBuffer(int port) const {
    if (outputTextures_.empty()) return nullptr;
    int width = (outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING;
    auto * out = createGPUBuffer(width, height_, PIXEL_PACKING, getOutputOrder(port), getOutputType(port), 0);
    pushSliceToBuffer(out, outputTextures_[0], width, height_, PIXEL_PACKING, getOutputType(port));
    return out;
}


/**
 * @copydoc GPULayerBase::getGPUInputBuffer
 */
GPUBuffer * CausalMultiHeadAttentionLayer::getGPUInputBuffer(int port) const {
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
BufferSpec::order CausalMultiHeadAttentionLayer::getInputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order CausalMultiHeadAttentionLayer::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void CausalMultiHeadAttentionLayer::updateFBOs() {
    // TODO (mw) when pooling is enabled this needs to be called on forward after the code below has been adjusted properly
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void CausalMultiHeadAttentionLayer::setupFBOs() {
    int fullqkvwidth = embedDim_ / PIXEL_PACKING;
    int fullqkvheight = height_;
    uint32_t scope1 = (context().texturePool()) ? context().texturePool()->scopeID() : 0;
    uint32_t scope2 = (context().texturePool()) ? context().texturePool()->scopeID() : 0;
    // ----------------------------------------------------------------
    // Textures and FBOs for Q, K and V computation. In case we have no
    // positional encoding step, we skip the textures and FBOs for
    // query and key and write them directly into the PE-stage buffers
    // ----------------------------------------------------------------
    if (posEnc_ != PosEncType::NONE) {
        keyTexture_ = Texture2D(fullqkvwidth, fullqkvheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope1, false);
        queryTexture_ = Texture2D(fullqkvwidth, fullqkvheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope2, false);     // NOTE (mw) this might be mapped to the same as keyTexture
    }
    valueTexture_ = Texture2D(fullqkvwidth, fullqkvheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope1, true);  // 32-bit
    qkvFBOs_.push_back((posEnc_ == PosEncType::NONE) ? nullptr : new FBO(context(), queryTexture_));
    qkvFBOs_.push_back((posEnc_ == PosEncType::NONE) ? nullptr : new FBO(context(), keyTexture_));
    qkvFBOs_.push_back(new FBO(context(), valueTexture_));
    // ----------------------------------------------------------------
    // Two FBOs for positional encoding, where the key FBO/texture is
    // used as cache when incremental mode is switched on
    // ----------------------------------------------------------------
    peQueryTexture_ = Texture2D(fullqkvwidth, fullqkvheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope2,  false);  // NOTE (mw) this might be mapped to the same as keyTexture
    peQueryFBO_ = new FBO(context(), peQueryTexture_);
    peKeyTexture_ = Texture2D(fullqkvwidth, fullqkvheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope1, true);
    peKeyFBO_ = new FBO(context(), peKeyTexture_);
    // ----------------------------------------------------------------
    // FBO for dot product implementations. Batched version will use a
    // single FBO with texture size defined by the maximum number of
    // sequence tokens. Note that usually the query length is not really
    // exhaustive, so on standard runs we may be able to stuff in multiple
    // batches (depending on the query length) in this texture. We use
    // the same texture for the single token version, as the max height
    // for that would be defined by the number of heads
    // ----------------------------------------------------------------
    int batcheddpwidth = height_;
    int batcheddpheight = std::max(height_, numHeads_ / PIXEL_PACKING);
    dotProdTexture_ = Texture2D(batcheddpwidth, batcheddpheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope2, false);
    dotProdFBO_ = new FBO(context(), dotProdTexture_);
    // ----------------------------------------------------------------
    // FBO for the softmax computations. Unsurprisingly, the size here
    // matches the size of the DP textures
    // ----------------------------------------------------------------
    smPass2BatchTexture_ = Texture2D(batcheddpwidth, batcheddpheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope2, false);
    smPass2BatchFBO_ = new FBO(context(), smPass2BatchTexture_);
    // ----------------------------------------------------------------
    // FBO for attention-weight/value multiply
    // ----------------------------------------------------------------
    attValTexture_ = Texture2D(fullqkvwidth, fullqkvheight, TEXTURE_PIXTYPE, 4, context().texturePool(), scope2, false);
    attValFBO_ = new FBO(context(), attValTexture_);
    // ----------------------------------------------------------------
    // FBO for the final projection. This is a single FBO that is wrapped
    // around the output texture which was supplied to this layer
    // ----------------------------------------------------------------
    assert(outputTextures_.size() == 1);
    framebuffers_.push_back(new FBO(context(), width_, height_, outputTextures_.at(0)));
    // ----------------------------------------------------------------
    // Setup some of the auxiliary functions here, some of them need the
    // scope ID
    // ----------------------------------------------------------------
    softMaxBatched_->setup(scope2);
    softMaxSingle_->setup(scope2);
    // ----------------------------------------------------------------
    // If we are not caching, unlock the textures here again
    // ----------------------------------------------------------------
    if (auto * pool = context().texturePool() ; (pool) && (!incremental_)) {
        pool->unlockTexture(peKeyTexture_);
        pool->unlockTexture(valueTexture_);
    }
}


/**
 * @brief Compute multi-head attention for the provided input
 *
 * @see RotaryEncoder, MatMulConst, SoftMaxBatched, SoftMaxSingle, AttentionMulBatched, AttentionMulSingle
 */
void CausalMultiHeadAttentionLayer::compute() {
    using mm = rudiments::MatMulConst;
    // --------------------------------------------------------
    // Initial computation of query, key and value matrices...
    // --------------------------------------------------------
    computeQKV();
    // --------------------------------------------------------
    // Dot-product (special case for single tokens and a batch
    // mode for multiple tokens)...
    // --------------------------------------------------------
    if (queryLength_ == 1) {
        // ----------------------------------------------------
        // Single stuff
        // ----------------------------------------------------
        dotProdSingle_->forward(peQueryFBO_->getAttachment(), peKeyFBO_->getAttachment(), keyLength_, dotProdFBO_);
        softMaxSingle_->forward(dotProdFBO_->getAttachment(), tokenIndex_, keyLength_, smPass2BatchFBO_);
        attMulSingle_->forward(qkvFBOs_.at(2)->getAttachment(), smPass2BatchFBO_->getAttachment(), tokenIndex_, keyLength_, attValFBO_);
    } else {
        // ----------------------------------------------------
        // Batched stuff
        // ----------------------------------------------------
        int head = 0;
        int batchsize = std::min(numHeads_ / PIXEL_PACKING, dpMaxHeadBatchSize_);
        do {
            dotProdBatched_->forward(peQueryFBO_->getAttachment(), peKeyFBO_->getAttachment(), queryLength_, keyLength_, head, batchsize, dotProdFBO_);
            softMaxBatched_->forward(dotProdFBO_->getAttachment(), tokenIndex_, queryLength_, keyLength_, batchsize, smPass2BatchFBO_);
            attMulBatched_->forward(qkvFBOs_[2]->getAttachment(), smPass2BatchFBO_->getAttachment(), queryLength_, tokenIndex_, head, batchsize, attValFBO_);
            int newhead = head + batchsize * PIXEL_PACKING;
            if (newhead > numHeads_) batchsize -= (newhead - numHeads_) / PIXEL_PACKING;
            head = newhead;
        } while (batchsize > 0 && head < numHeads_);
    }
    // --------------------------------------------------------
    // Output projection
    // --------------------------------------------------------
    assert(outMul_);
    assert(!framebuffers_.empty());
    glActiveTexture(GL_TEXTURE0 + mm::INPUT0_UNIT);
    glBindTexture(GL_TEXTURE_2D, attValFBO_->getAttachment());
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        glActiveTexture(GL_TEXTURE0 + mm::RESIDUAL_UNIT);
        glBindTexture(GL_TEXTURE_2D, residualTextures_.at(0));
    }
    outMul_->forward(queryLength_, 0, framebuffers_.at(0));
}


/**
 * @brief Compute Q, K and V tensors
 */
void CausalMultiHeadAttentionLayer::computeQKV() {
    // There is missing handling of wraparounds. Whenever the token index + the number of tokens
    // exceeds the height of the texture, it should wrap around and overwrite at row 0 which makes
    // it a bit complicated in case of a wraparound with multiple tokens at once
    // -> it might be a better idea to do the wraparound in the engine by splitting the
    //    query appropriately
    using mm = rudiments::MatMulConst;
    assert(queryMul_);
    assert(keyMul_);
    assert(valueMul_);
    // --------------------------------------------------------
    // Compute query items, note that those are never cached
    // --------------------------------------------------------
    glActiveTexture(GL_TEXTURE0 + mm::INPUT0_UNIT);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    queryMul_->forward(queryLength_, 0, (posEnc_ == PosEncType::NONE) ? peQueryFBO_ : qkvFBOs_.at(0));
    if (posEnc_ == PosEncType::ROTARY) {
        rotaryEncoder_->forward(qkvFBOs_.at(0)->getAttachment(), tokenIndex_, queryLength_, 0, peQueryFBO_);
    }
    // --------------------------------------------------------
    // Compute key items, these will be cached in an incremental
    // decoding scenario...
    // --------------------------------------------------------
    glActiveTexture(GL_TEXTURE0 + mm::INPUT0_UNIT);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    if (posEnc_ == PosEncType::NONE) {
        keyMul_->forward(queryLength_, tokenIndex_, peKeyFBO_);
    } else {
        keyMul_->forward(queryLength_, 0, qkvFBOs_.at(1));
    }
    if (posEnc_ == PosEncType::ROTARY) {
        rotaryEncoder_->forward(qkvFBOs_.at(1)->getAttachment(), tokenIndex_, queryLength_,
                                tokenIndex_, peKeyFBO_);
    }
    // --------------------------------------------------------
    // Compute value items, these will be cached in an incremental
    // decoding scenario...
    // --------------------------------------------------------
    glActiveTexture(GL_TEXTURE0 + mm::INPUT0_UNIT);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    valueMul_->forward(queryLength_, (incremental_) ? tokenIndex_ : 0,  qkvFBOs_.at(2));
    keyLength_ = (incremental_) ? (tokenIndex_ + queryLength_) : queryLength_;
}


} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
