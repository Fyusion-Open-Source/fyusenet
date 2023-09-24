//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Attention DP / Value Multiplication - Single Tokens (Header)                (c) Martin Wawro 2023
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
class SequenceTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence::rudiments {

/**
 * @brief Compute matrix-product of attention weights and values for a single token
 *
 * This class computes the matrix product of a single-token query weights and the attention values
 * (V matrix) for a subset of heads.
 *
 * \include attmul_single_appendix.inc
 */
class AttentionMulSingle : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SequenceTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    AttentionMulSingle(int width, int numHeads, int headDim, const GfxContextLink& ctx);
    ~AttentionMulSingle() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup();
    void forward(GLuint valueTexture, GLuint smTexture, int tokenIndex, int keyLength, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_;                               //!< Pixel width of the embedding dimension of a sequence
    int numHeads_ = 0;                        //!< Number of attention heads
    int headDim_ = 0;                         //!< Dimensionality of each attention head (in atoms)
    int maxSingleWeights_ = 0;                //!< Maximum weights that can be processed in a single pass
    opengl::VAO * array_ = nullptr;           //!< Vertex array object for proxy geometry
    opengl::VBO * vertices_ = nullptr;        //!< Vertex buffer for proxy geometry
    opengl::programptr shader_;               //!< Shader program that performs the computation
    constexpr static int USED_VARYINGS = 3;   //!< Number of varying parameters used for basic shader functions
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

