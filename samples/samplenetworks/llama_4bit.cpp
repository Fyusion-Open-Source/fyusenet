//--------------------------------------------------------------------------------------------------
// FyuseNet Samples
//--------------------------------------------------------------------------------------------------
// LLaMa Generative Language Model Sample                                      (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


#define DOWNLOAD_DATA

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "llama_4bit.h"
#include "../helpers/llama_4bit_params.h"
#include <fyusenet/common/miscdefs.h>
#include <fyusenet/gpu/custom/sequence/linear_hadamard.h>

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 */
LlaMa4Bit::LlaMa4Bit(const fyusion::fyusenet::GfxContextLink& context) : NeuralNetwork(context) {
}


/**
 * @brief Destructor
 *
 * Deallocates resources
 */
LlaMa4Bit::~LlaMa4Bit() {
    FNET_DEL_AND_CLEAR(cpuTokenOut_);
    FNET_DEL_AND_CLEAR(gpuTokenOut_);
    for (int i=0; i < ASYNC_BUFFERS; i++) {
        delete inBuffers_[i];
        inBuffers_[i] = nullptr;
    }
}

/**
 * @copydoc NeuralNetwork::cleanup
 */
void LlaMa4Bit::cleanup() {
    FNET_DEL_AND_CLEAR(gpuTokenOut_);
    NeuralNetwork::cleanup();
}


/**
 * @brief Run inference on one or multiple tokens
 *
 * @param state Pointer to StateToken object that provides information about the token(s)
 *
 * @return Execution state of network run
 *
 * @see NeuralNetwork::forward
 */
fyusion::fyusenet::NeuralNetwork::execstate LlaMa4Bit::forward(fyusion::fyusenet::StateToken *state) {
    assert(state);
    if (!uploadRequired_) {
        state->maskLayers.insert(1);
    } else state->maskLayers.clear();
    return NeuralNetwork::forward(state);
}


/**
 * @brief Set input token(s) for the network
 *
 * @param tokens Pointer to array of tokens
 * @param numTokens Number of tokens in \p tokens, must be at least 1 and smaller than maximum
 *                  sequence length
 *
 * Copies the supplied tokens into an internal upload buffer, which will be uploaded to the GPU
 * on the next call to forward().
 */
void LlaMa4Bit::setInputTokens(const uint32_t * tokens, int numTokens) {
    using namespace fyusion::fyusenet;
    // TODO (mw) multi-threading support
    const int numbuffers = 1;
    assert(tokens);
    assert(numTokens > 0);
    assert(numTokens < maxSequenceLen_);
    // -------------------------------------------------------
    // Make sure that we have the necessary amount of buffers
    // allocated...
    // -------------------------------------------------------
    for (int i=0; i < numbuffers; i++) {
        if (!inBuffers_[i]) {
            inBuffers_[i] = new cpu::CPUBuffer(BufferShape(maxSequenceLen_, 1, 1, 0, BufferShape::type::UINT32, BufferSpec::order::GPU_SEQUENCE));
        }
    }
    gpu::UploadLayer * upload = dynamic_cast<gpu::UploadLayer *>(engine_->getLayers()["upload"]);
    assert(upload);
    CPUBuffer * buf = inBuffers_[0];
    auto * tgt = buf->map<uint32_t>();
    assert(tgt);
    memcpy(tgt, tokens, numTokens * sizeof(uint32_t));
    buf->unmap();
    upload->setCPUInputBuffer(buf, 0);
    auto * embedding = dynamic_cast<gpu::GPULayerBase *>(engine_->getLayers()["embedding"]);
    auto * gbuf = upload->getGPUOutputBuffer(0);
    embedding->setGPUInputBuffer(gbuf, 0);
    delete gbuf;
    uploadRequired_ = true;
}


/**
 * @brief Internally rotate the output token back to the input token
 *
 * This function directly rotates the output token of the token-scoring layer to be the next input
 * token for the following network run (without downloading/uploading) the token first.
 */
void LlaMa4Bit::rotateInputToken() {
    using namespace fyusion::fyusenet;
    auto * embedding = dynamic_cast<gpu::GPULayerBase *>(engine_->getLayers()["embedding"]);
    assert(embedding);
    embedding->setGPUInputBuffer(gpuTokenOut_, 0);
    uploadRequired_ = false;
}


/**
 * @brief Set a parameter file to use for loading network parameters like weights and biases
 *
 * @param filename Filename to use
 */
void LlaMa4Bit::useParameterFile(const std::string &filename) {
    fileParameters_.reset(new LlaMa4BitFileParameters(filename));
}


/**
 * @brief Retrieve the predicted token index from the download layer that follows the token scoring
 *
 * @return 32-bit integer index into token list, or \c ILLEGAL_TOKEN if something went wront
 */
uint32_t LlaMa4Bit::getPredictedToken() const {
    if (cpuTokenOut_) {
        uint32_t token = ILLEGAL_TOKEN;
        cpuTokenOut_->with<uint32_t>([&](const uint32_t * ptr) {
            if (ptr) token = *ptr;
        });
        return token;
    }
    return ILLEGAL_TOKEN;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc NeuralNetwork::initializeWeights
 */
void LlaMa4Bit::initializeWeights(fyusion::fyusenet::CompiledLayers & layers) {
    assert(fileParameters_);
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        it.second->loadParameters(fileParameters_.get());
    }
}


/**
 * @copydoc NeuralNetwork::buildLayers
 */
fyusion::fyusenet::CompiledLayers LlaMa4Bit::buildLayers() {
    using namespace fyusion::fyusenet;
    std::shared_ptr<LayerFactory> factory = getLayerFactory();
    auto * upload = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::UPLOAD, "upload");
    upload->context(context()).shape(1, 1, 1).sequence(maxSequenceLen_).dataType(BufferSpec::dtype::UINT32).sequencePacking(1).number(layerNo_++);
    upload->push(factory);
    auto * embbld = new gpu::EmbeddingLayerBuilder("embedding");
    embbld->context(context()).sequence(maxSequenceLen_).outChannels(embedDim_).tableRows(vocabularySize_).number(layerNo_++);
    embbld->push(factory);
    for (int i=0; i < numDecoderBlocks_; i++) buildDecoderBlock(factory, i);
    auto * normbld = new gpu::GPULayerBuilder("modelnorm");
    normbld->sequence(maxSequenceLen_).channels(embedDim_).type(LayerType::RMSNORM).context(context()).number(layerNo_++);
    normbld->push(factory);
    auto * scorebld = new fyusion::fyusenet::gpu::TokenScoringLayerBuilder("tokenscoring");
    scorebld->context(context()).sequence(maxSequenceLen_).inChannels(embedDim_).outChannels(1).tableRows(vocabularySize_).number(layerNo_++);
    scorebld->push(factory);
#ifdef DOWNLOAD_DATA
    auto * downbld = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::DOWNLOAD, "download");
    downbld->context(context()).shape(1, 1, 1).sequence(maxSequenceLen_).dataType(BufferSpec::dtype::UINT32).sequencePacking(1).number(layerNo_++);
    downbld->push(factory);
#endif
    return factory->compileLayers();
}


/**
 * @copydoc NeuralNetwork::connectLayers
 */
void LlaMa4Bit::connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::opengl;
    int idx = 1;
    buffers->connectLayers(layers[idx], layers[idx+1], 0);   // upload -> embedding
    idx++;
    for (int i = 0; i < numDecoderBlocks_; i++) idx = connectDecoderBlock(layers, buffers, idx);
    buffers->connectLayers(layers[idx], layers[idx+1], 0);   // last down projection -> modelnorm
    idx++;
    buffers->connectLayers(layers[idx], layers[idx+1], 0);   // modelnorm -> score
    idx++;
#ifdef DOWNLOAD_DATA
    buffers->connectLayers(layers[idx], layers[idx+1], 0);   // score -> download
    gpuTokenOut_ = dynamic_cast<gpu::GPULayerBase *>(layers[idx])->getGPUOutputBuffer(0);
    auto * down = dynamic_cast<gpu::DownloadLayer *>(layers[idx+1]);
    assert(down);
    auto shape = down->getOutputShape(0);
    cpuTokenOut_ = shape.createCPUBuffer();
    down->addCPUOutputBuffer(cpuTokenOut_, 0);
#else
    gpuTokenOut_ = gpu::GPUBuffer::createSequenceBuffer(BufferShape(1, maxSequenceLen_, BufferShape::type::UINT32, 1), true);
    auto * score = dynamic_cast<gpu::GPULayerBase *>(layers[idx]);
    score->setGPUOutputBuffer(gpuTokenOut_, 0);
#endif
}


/**
 * @brief Establish connections within a single decoder block
 *
 * @param layers Reference to set of compiled layers
 * @param buffers Pointer to BufferManager instance
 * @param startIndex Index of first layer in the block
 *
 * @return Index of the last layer in the block that haas been connected
 */
int LlaMa4Bit::connectDecoderBlock(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers, int startIndex) {
    buffers->connectLayers(layers[startIndex], layers[startIndex + 1], 0);       // in -> ln0
    buffers->connectLayers(layers[startIndex + 1], layers[startIndex + 2], 0);   // ln0 -> att
    buffers->connectLayers(layers[startIndex], layers[startIndex + 2], 1);       // in -> att (residual)
    buffers->connectLayers(layers[startIndex + 2], layers[startIndex + 3], 0);   // att -> ln1
    buffers->connectLayers(layers[startIndex + 3], layers[startIndex + 4], 0);   // ln1 -> gate
    buffers->connectLayers(layers[startIndex + 3], layers[startIndex + 5], 0);   // ln1 -> up
    buffers->connectLayers(layers[startIndex + 4], layers[startIndex + 6], 0);   // gate -> down
    buffers->connectLayers(layers[startIndex + 5], layers[startIndex + 6], 1);   // up -> down
    buffers->connectLayers(layers[startIndex + 2], layers[startIndex + 6], 2);   // att -> down (residual)
    return startIndex + 6;
}


/**
 * @brief Build a single decoder block
 *
 * @param factory LayerFactory instance to use for layer building
 * @param blockNum Decoder layer block number (0 for the first decoder block and so on)
 *
 * This function sets up a set of layer builders and passes them to the layer factory for later
 * instantiation. The following layers are created:
 *  1. Input layer-norm (RMS)
 *  2. Causally-masked multi-head attention
 *  3. Post-attention layer-norm (RMS)
 *  4. MLP part, consisting of gate layer, up and down layer
 */
void LlaMa4Bit::buildDecoderBlock(std::shared_ptr<fyusion::fyusenet::LayerFactory> & factory, int blockNum) {
    using namespace fyusion::fyusenet;
    char name[256];
    //-------------------------------------------------
    // Input layer-norm (RMS)
    //-------------------------------------------------
    snprintf(name, sizeof(name), "dec%dln0", blockNum);
    auto * ln0bld = new gpu::GPULayerBuilder(name);
    ln0bld->sequence(maxSequenceLen_).channels(embedDim_).type(LayerType::RMSNORM).context(context()).number(layerNo_++);
    ln0bld->push(factory);
    //-------------------------------------------------
    // Causally-masked multi-head attention
    //-------------------------------------------------
    snprintf(name, sizeof(name), "dec%datt", blockNum);
    auto * attbld = new gpu::AttentionLayerBuilder(name);
    attbld->sequence(maxSequenceLen_).channels(embedDim_).heads(numHeads_).headDim(headDim_).
        quantize(qt_type::QT_MIXED_FLOAT, param_type::WGT_INT4).quantGroupSize(quantGroupSize_).
        positionalEncoding(PosEncType::ROTARY).rotaryThetaBase(thetaBase_).incremental().residual().
        causal().context(context()).number(layerNo_++);
    attbld->push(factory);
    //-------------------------------------------------
    // Post-attention layer-norm (RMS)
    //-------------------------------------------------
    snprintf(name, sizeof(name), "dec%dln1", blockNum);
    auto * ln1bld = new gpu::GPULayerBuilder(name);
    ln1bld->sequence(maxSequenceLen_).channels(embedDim_).type(LayerType::RMSNORM).context(context()).number(layerNo_++);
    ln1bld->push(factory);
    //-------------------------------------------------
    // MLP part, gate layer...
    //-------------------------------------------------
    snprintf(name, sizeof(name), "dec%dgate", blockNum);
    auto * gatebld = new gpu::LinearLayerBuilder(name);
    gatebld->context(context()).quantize(qt_type::QT_MIXED_FLOAT, param_type::WGT_INT4).
        quantGroupSize(quantGroupSize_).sequence(maxSequenceLen_).
        inChannels(embedDim_).outChannels(mlpIntermediate_).number(layerNo_++);
    gatebld->push(factory);
    //-------------------------------------------------
    // Up projection
    //-------------------------------------------------
    snprintf(name, sizeof(name), "dec%dup", blockNum);
    auto * upbld = new gpu::LinearLayerBuilder(name);
    upbld->context(context()).quantize(qt_type::QT_MIXED_FLOAT, param_type::WGT_INT4).
            quantGroupSize(quantGroupSize_).sequence(maxSequenceLen_).
            inChannels(embedDim_).outChannels(mlpIntermediate_).number(layerNo_++);
    upbld->push(factory);
    //-------------------------------------------------
    // Down projection (w/ Hadamard product)
    //-------------------------------------------------
    snprintf(name, sizeof(name), "dec%ddown", blockNum);
    auto * dwnbld = gpu::custom::sequence::LinearHadamardLayer::createBuilder(name, qt_type::QT_MIXED_FLOAT, param_type::WGT_INT4, quantGroupSize_, false);
    dwnbld->context(context()).sequence(maxSequenceLen_).inChannels(mlpIntermediate_).outChannels(embedDim_).
    prefixAct(ActType::SILU, 1).residual().number(layerNo_++);
    dwnbld->push(factory);
}


