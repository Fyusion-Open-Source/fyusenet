//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Multi-Head Causally-Masked Self-Attention Layer (Header)                    (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/texture.h"
#include "../../base/bufferspec.h"
#include "../../base/parameterprovider.h"
#include "../uniformweightarray.h"
#include "../gpulayerbase.h"
#include "../attentionlayerbuilder.h"
#include "../deep/deeptiler.h"
#include "../sequence/rudiments/rotary_encoding.h"
#include "../sequence/rudiments/masked_softmax_batched.h"
#include "../sequence/rudiments/masked_softmax_single.h"
#include "../sequence/rudiments/attmul_batched.h"
#include "../sequence/rudiments/attmul_single.h"
#include "../sequence/rudiments/dotprod_batched.h"
#include "../sequence/rudiments/dotprod_single.h"
#include "../sequence/rudiments/matmul_const.h"

class AttentionTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence {

/**
 * @brief Compound layer that performs causally masked multi-head attention
 *
 * This layer performs a series of internal computations to compute the output of a multi-head
 * attention layer which uses implicit causal masking. The layer expects a token-embedding
 * sequence as input and produces a sequence of token embeddings as output. The input texture
 * format is given by a simple row-wise concatenation of the token embeddings, for example:
 *
  * @code
 *  +--------------------------------------------------------------------+
 *  |  e0  |  e1  | e2  |  e3  |  e4  |  e5  |  e6  |     ...   |  e<d>  | token 0
 *  +--------------------------------------------------------------------+
 *  |  e0  |  e1  | e2  |  e3  |  e4  |  e5  |  e6  |     ...   |  e<d>  | token 1
 *  +--------------------------------------------------------------------+
 *  |                          .............                             | token ...
 *  +--------------------------------------------------------------------+
 *  |  e0  |  e1  | e2  |  e3  |  e4  |  e5  |  e6  |     ...   |  e<d>  | token N
 *  +--------------------------------------------------------------------+
 * @endcode
 *
 * As is common in attention layers, the input is first linearly transformed into three parts:
 * query \f$ \mathbf{Q} \f$, key \f$ \mathbf{K} \f$ and value \f$ \mathbf{V} \f$.
 *
 * The resulting matrices are now interpreted differently, since the linear operators transformed
 * them from their original embedding space into multiple smaller subspaces called \e heads.
 * Another interpretation is that the embedding vector is first split into subspaces and then
 * transformed individually on each subspace, which amounts to the same thing implementation-wise.
 * The format of the Q, K and V tensors is given by (example for 32 heads with each head having a
 * cardinality of 128):
 *
 * @code
 *  32 (head_dim/4) 32 (head_dim/4)                           32 (head-dim/4)
 * +---------------+---------------+-----------------------+------------------+
 * |  T0(0) head0  | T0(32) head1  | ......................| T0(1023) head 32 |
 * +---------------+---------------+-----------------------+------------------+
 * |  T1(0) head0  | T1(32) head1  | ......................| T1(1023) head 32 |
 * +---------------+---------------+-----------------------+------------------+
 * |      ...      |      ...      |         ...           |      ...         |
 * +---------------+---------------+-----------------------+------------------+
 * |  Tk(0) head0  | Tk(32) head1  | ......................| Tk(1023) head 32 |
 * +---------------+---------------+-----------------------+------------------+
 *  RGBA RGBA ....  RGBA RGBA ....        ....               RGBA ....    RGBA
 * @endcode
 *
 * The transformed tensors are then used to compute the attention weights \f$ \mathbf{A} \f$:
 *
 * \f[ \mathbf{A} = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \f]
 *
 * where - again - technically this computation is done for every head independently. A causal mask
 * is applied to the attention weights to ensure that the attention is only computed for tokens that
 * are temporally \e before the current token, this is all done implicitly in the computation.
 *
 * The attention weights are then used to multiply the value tensor \f$ \mathbf{V} \f$:
 *
 * \f[ \mathbf{O} = \mathbf{A} \mathbf{V} \f]
 *
 * and finally the resulting output tensor \f$ \mathbf{O} \f$ is linearly transformed back into
 * an embedding space:
 *
 * \f[ \mathbf{E} = \mathbf{O} \mathbf{W} \f]
 *
 * This class supports an optional positional encoding step after the initial computation of
 * \f$ \mathbf{Q}, \mathbf{K}, \mathbf{V} \f$ which is applied to the query tensor \f$ \mathbf{Q} \f$
 * and the key tensor \f$ \mathbf{K} \f$ only prior to the dot-product computation.
 *
 * @warning This layer only supports 4-bit quantized weights as of now. It is also largely untested
 *          \e without the positional encoding step.
 */
class CausalMultiHeadAttentionLayer : public gpu::GPULayerBase {
    friend class ::AttentionTest;
    using RotaryEncoder = rudiments::RotaryEncoder;
    using MaskedSoftMaxBatched = rudiments::MaskedSoftMaxBatched;
    using MaskedSoftMaxSingle = rudiments::MaskedSoftMaxSingle;
    using AttentionMulBatched = rudiments::AttentionMulBatched;
    using AttentionMulSingle = rudiments::AttentionMulSingle;
    using DotProductBatched = rudiments::DotProductBatched;
    using DotProductSingle = rudiments::DotProductSingle;
    using MatMulConst = rudiments::MatMulConst;

 public:

    enum mtxid : uint8_t {
        QUERY = 0,
        KEY,
        VALUE,
        OUTPUT,
        NUM_MATRICES
    };

    constexpr static int MAX_DP_BATCH = 8;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    CausalMultiHeadAttentionLayer(const AttentionLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void setup() override;
    void cleanup() override;
    void loadParameters(const ParameterProvider * weights) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] GPUBuffer *getGPUOutputBuffer(int port) const override;
    [[nodiscard]] GPUBuffer *getGPUInputBuffer(int port) const override;
    void writeResult(const char *fileName, bool includePadding) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;
    void setupFBOs() override;
    void updateFBOs() override;
    void compute();
    void computeQKV();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    Texture2D queryTexture_;                            //!< Non-caching texture for the queries
    Texture2D keyTexture_;                              //!< (Possibly) caching texture for the keys
    Texture2D valueTexture_;                            //!< Caching texture for the values
    Texture2D dotProdTexture_;                          //!< Texture that holds the result of the dot-product computation (see #dotProdFBO_)
    Texture2D smPass2BatchTexture_;                     //!< Texture that holds the result of the batched softmax computation
    Texture2D attValTexture_;                           //!< Texture that holds the result of the attention-weighted projection of the values
    Texture2D peQueryTexture_;                          //!< Non-caching texture for the positions-encoded queries
    Texture2D peKeyTexture_;                            //!< (Possibly) caching texture for the position-encoded keys
    FBO * peQueryFBO_ = nullptr;                        //!< FBO that wraps the positional encoding of the query
    FBO * peKeyFBO_ = nullptr;                          //!< FBO that wraps the positional encoding of the key (cached for autoregressive / incremental use), see #keyTexture_
    FBO * dotProdFBO_ = nullptr;                        //!< FBO that wraps the result of the dot-product computation (see #dotProdTexture_)
    FBO * smPass2BatchFBO_ = nullptr;                   //!< FBO that wraps the batched softmax computation results (see #smPass2BatchTexture_)
    FBO * attValFBO_ = nullptr;                         //!< FBO that wraps the attention-weighted projection of the values (see #attValTexture_)
    std::vector<FBO *> qkvFBOs_;                        //!< FBOs that hold the Q, K and V tensors
    int dpMaxHeadBatchSize_ = MAX_DP_BATCH;             //!< Maximum batch size for the dot-product computation
    int numHeads_ = 0;                                  //!< Total number of attention heads
    int headDim_ = 0;                                   //!< Dimension of a single attention head
    uint16_t embedDim_ = 0;                             //!< Dimension of the embedding space
    uint16_t maxSequenceLength_ = 0;                    //!< Maximum sequence length supported by the model
    int queryLength_ = 0;                               //!< Number of query tokens in the sequence
    int keyLength_ = 0;                                 //!< Number of keys in the context (including cached)
    int tokenIndex_ = 0;                                //!< Current token index supplied to the layer (for incremental mode)
    int quantGroupSize_ = 0;                            //!< Number of weights per quantization group
    bool incremental_ = false;                          //!< Whether the layer is operating in incremental mode
    bool autoResidual_ = false;                         //!< Whether the layer operates in a mode where it adds its output to the input automatically
    PosEncType posEnc_ = PosEncType::NONE;              //!< Type of positional encoding used by the layer
    MatMulConst * queryMul_ = nullptr;                  //!< Pointer to query multiplication instance
    MatMulConst * keyMul_ = nullptr;                    //!< Pointer to key computation instance
    MatMulConst * valueMul_ = nullptr;                  //!< Pointer to value computation instance
    MatMulConst * outMul_ = nullptr;                    //!< Pointer to final output projection instance
    bool hasBias_ = false;                              //!< Indicator whether any projection used inside the layer has a bias / is affine  (TODO support this)
    RotaryEncoder * rotaryEncoder_ = nullptr;           //!< Pointer to rotary encoder instance
    MaskedSoftMaxBatched * softMaxBatched_ = nullptr;   //!< Pointer to batched softmax instance
    MaskedSoftMaxSingle * softMaxSingle_ = nullptr;     //!< Pointer to single softmax instance
    AttentionMulBatched * attMulBatched_ = nullptr;     //!< Pointer to batched attention multiplication instance
    AttentionMulSingle * attMulSingle_ = nullptr;       //!< Pointer to single attention multiplication instance
    DotProductBatched * dotProdBatched_ = nullptr;      //!< Pointer to batched dot-product instance
    DotProductSingle * dotProdSingle_ = nullptr;        //!< Pointer to single dot-product instance

    /**
     * Type of quantization to be used in computation
     */
    qt_type quantType_ = qt_type::QT_NONE;

    /**
     * Data type for the weights supplied to this layer
     */
    param_type dataType_ = param_type::WGT_FLOAT;
};

} // fyusion::fyusenet::gpu::sequence namespace


// vim: set expandtab ts=4 sw=4:
