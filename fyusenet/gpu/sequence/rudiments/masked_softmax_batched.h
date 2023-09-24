//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Masked Softmax Operation (Header)                                           (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


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
class SoftMaxTest;

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::sequence::rudiments {

/**
 * @brief Causally-masked SoftMax operation for multiple query tokens
 *
 * This class computes a causally-masked softmax operator on an input texture that contains the
 * \e batched dot-product result of a multi-head attention layer. It is to be used in a batch-loop
 * over the head dimension, that runs in the form of:
 *   1. dot product
 *   2. softmax
 *   3. attention/value multiplication
 *
 * Be aware that this class does not just compute the softmax of the input, but rather a \e masked
 * softmax, where results from key-tokens that are outside the scope of the corresponding query-token
 * index are set to zero (causal masking).
 *
 * The computation done here takes place in two passes: first the denominators are computed and
 * stored in an intermediary texture and then in a 2nd pass the actual (masked) softmax is
 * computed.
 *
 * \include msm_batch_appendix.inc
 */
class MaskedSoftMaxBatched : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SoftMaxTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    MaskedSoftMaxBatched(int maxSeq, int maxBatch, const GfxContextLink& ctx);
    ~MaskedSoftMaxBatched() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup(uint32_t texturePoolScope);
    void forward(GLuint srcTexture, int tokenIndex, int numTokens, int keyLength, int batchSize, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int maxSeqLen_ = 0;
    int innerBatchSize_ = 16;
    int maxBatch_ = 0;
    opengl::VAO * pass1Array_ = nullptr;
    opengl::VBO * pass1Vertices_ = nullptr;
    opengl::VAO * pass2Array_ = nullptr;
    opengl::VBO * pass2Vertices_ = nullptr;
    opengl::IBO * pass2Indices_ = nullptr;
    opengl::programptr pass1Shader_;
    opengl::programptr pass2Shader_;
    opengl::FBO * pass1FBO_ = nullptr;
    opengl::Texture2D pass1Texture_;
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

