//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Attention Weights / Value Multiplication - Multi Tokens (Header)            (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../../gl/gl_sys.h"
#include "../../../gl/shaderprogram.h"
#include "../../../gl/uniformstate.h"
#include "../../../gl/fbo.h"
#include "../../../gl/vao.h"
#include "../../../gl/vbo.h"
#include "../../../gl/ibo.h"
#include "../../../gl/ubo.h"
#include "../../../gl/texture.h"

class AttentionTest;
class SequenceTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence::rudiments {

/**
 * @brief Compute matrix-product of attention weights and values for a set of tokens
 *
 * This class computes the matrix product of a batch of attention weights and the attention values
 * (V matrix) for a subset of heads. It runs as part of a batch-loop which splits the computation
 * of the dot-product, softmax and matrix product into batches along the head-dimension in order to
 * save texture memory.
 *
 * \include attmul_batch_appendix.inc
 */
class AttentionMulBatched : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SequenceTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    AttentionMulBatched(int numHeads, int headDim, int maxSeq, const GfxContextLink& ctx);
    ~AttentionMulBatched() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup();
    void forward(GLuint valueTexture, GLuint smTexture, int numTokens, int tokenIndex, int headOffset, int batchSize, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int numHeads_ = 0;                      //!< Number of heads in the multi-head attention
    int headDim_ = 0;                       //!< Dimension of the attention heads
    int maxSequenceLength_ = 0;             //!< Maximum supported sequence length (must be allocated in the textures already)
    int maxBatchedWeights_ = 0;             //!< Maximum number of weights that can be batched in a single pass
    opengl::VAO * array_ = nullptr;         //!< Proxy geometry VAO
    opengl::VBO * vertices_ = nullptr;      //!< Proxy geometry VBO
    opengl::programptr shader_;             //!< Shader program that performs the computation
    std::vector<uint16_t> lines_;           //!< Offsets for the proxy geometry, which consists of a set of lines
    constexpr static int USED_VARYINGS = 3; //!< Number of varying parameters used for basic shader functions
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

