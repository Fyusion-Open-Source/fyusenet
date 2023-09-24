//--------------------------------------------------------------------------------------------------
// FyuseNet Samples
//--------------------------------------------------------------------------------------------------
// LLaMa Generative Language Model Sample (Header)                             (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <unordered_map>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include <fyusenet/gl/gl_sys.h>

//------------------------------------- Public Declarations ----------------------------------------

class LlaMa4BitFileParameters;

/**
 * @brief Llama-type language model using 4-bit mixed-precision FP quantization
 *
 * This class implements a LLaMa-type language model using 4-bit mixed-precision floating-point
 * quantization. As is common for LLM type models, this model uses a transformer architecture
 * with causally-masked self-attention layers and is fed with tokens, resulting in a new token
 * being predicted as output token which is then fed back into the model in an autoregressive
 * manner.
 *
 * @see https://github.com/facebookresearch/llama
 */
class LlaMa4Bit : public fyusion::fyusenet::NeuralNetwork {

    constexpr static int ASYNC_BUFFERS = 2;         // NOTE (mw) the async support for this net is not done yet

 public:
    constexpr static uint32_t ILLEGAL_TOKEN = 0xFFFFFFFF;
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit LlaMa4Bit(const fyusion::fyusenet::GfxContextLink& context = fyusion::fyusenet::GfxContextLink());
    ~LlaMa4Bit() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void useParameterFile(const std::string & filename);
    void cleanup() override;
    void setInputTokens(const uint32_t * tokens, int numTokens);
    void rotateInputToken();
    fyusion::fyusenet::NeuralNetwork::execstate forward(fyusion::fyusenet::StateToken *token) override;

    [[nodiscard]] uint32_t getPredictedToken() const;

    [[nodiscard]] int maxSequenceLen() const {
        return maxSequenceLen_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    fyusion::fyusenet::CompiledLayers buildLayers() override;
    void initializeWeights(fyusion::fyusenet::CompiledLayers & layers) override;
    void connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) override;
    void buildDecoderBlock(std::shared_ptr<fyusion::fyusenet::LayerFactory> & factory, int blockNum);
    [[nodiscard]] static int connectDecoderBlock(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers, int startIndex);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::unique_ptr<LlaMa4BitFileParameters> fileParameters_;
    int layerNo_ = 1;                                               //!< Tracking number for layer numbers / layer identification
    int numDecoderBlocks_ = 32;                                     //!< Number of total decoder blocks in the network
    int maxSequenceLen_ = 1024;                                     //!< Maximum number of tokens in the sequence
    int embedDim_ = 4096;
    int mlpIntermediate_ = 11008;
    int numHeads_ = 32;
    int headDim_ = 128;
    int quantGroupSize_ = 128;
    int vocabularySize_ = 32000;
    float thetaBase_ = 10000.f;
    bool uploadRequired_ = true;
    fyusion::fyusenet::gpu::GPUBuffer * gpuTokenOut_ = nullptr;
    fyusion::fyusenet::cpu::CPUBuffer * cpuTokenOut_ = nullptr;
    std::vector<int> attentionBlocks_;
    std::vector<int> mlpBlocks_;

    /**
     * Pointers to input token buffers (from the CPU)
     *
     * @see setInputTokens
     */
    fyusion::fyusenet::cpu::CPUBuffer * inBuffers_[ASYNC_BUFFERS] = {nullptr};
};

// vim: set expandtab ts=4 sw=4:
