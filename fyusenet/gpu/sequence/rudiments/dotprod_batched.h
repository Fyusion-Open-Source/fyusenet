//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Query / Key Dot-Product Computation - Multi Tokens (Header)                 (c) Martin Wawro 2023
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
 * @brief Compute dot product between query and key vectors for multiple query tokens in multi-head attention
 *
 * This class provides a shader interface for computing the dot-product between a collection of query
 * vectors and a collection of key vectors. Both, the query and key vectors are supplied as textures
 * and the result will be written to a target FBO <i>in batches</i> along the head dimension, to be
 * part of a batch-centric  computation in the form of:
 *   1. dot product
 *   2. softmax
 *   3. attention-value multiplication
 *
 * The reason for the batched approach is the prohibitively large size of a texture that can store
 * a dot product of large amounts query/key tokens. For example, when encountering a large
 * query/key sequence of 4096 tokens using 32 attention heads, the dot product would require a
 * texture of size 4096 x 4096 x 32, when using 16-bit floating point values to store that, the
 * texture would take up 1 GiB of GPU memory. With the batched approach we incrementally
 * compute the results for batches of heads, lowering the required texture size. In the example
 * above we would use 256 MiB of GPU memory instead.
 *
 * \include dp_batch_appendix.inc
 */
class DotProductBatched : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SequenceTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DotProductBatched(int numHeads, int headDim, int maxBatch, const GfxContextLink& ctx);
    ~DotProductBatched() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup();
    void forward(GLuint queryTexture, GLuint keyTexture, int numTokens, int keyLength, int headOffset, int batchSize, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int numHeads_ = 0;
    int headDim_ = 0;
    int innerBatchSize_ = 4;
    int maxBatch_ = 0;
    opengl::VAO * array_ = nullptr;
    opengl::VBO * vertices_ = nullptr;
    opengl::IBO * indices_ = nullptr;
    opengl::programptr shader_;
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

