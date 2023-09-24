//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Query / Key Dot-Product Computation - Single Token  (Header)                (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <memory>

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
 * @brief Compute dot product between query and key vectors for a single token in multi-head attention
 *
 * This class provides a shader interface for computing the dot-product between a single query
 * vector and a collection of key vectors. Both, the query and key vector are supplied as textures
 * and the result will be written to a target FBO on a per-head basis.
 *
 * When attention is used in an autoregressive manner (adding new tokens to the input incrementally),
 * a regular case for the dot-product computation consists of computing the dot-product
 * of the query token with the key tokens, having a single query token on one left-hand-side and
 * multiple key tokens on the right-hand-side.
 *
 * \include dp_single_appendix.inc
 */
class DotProductSingle : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SequenceTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DotProductSingle(int width, int numHeads, int headDim, const GfxContextLink& ctx);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup();
    void forward(GLuint queryTexture, GLuint keyTexture, int keyLength, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;                           //!< Width of the input query and key textures
    int numHeads_ = 0;                        //!< Number of heads in the multi-head attention
    int headDim_ = 0;                         //!< Dimension of the attention heads
    int innerBatchSize_ = 4;                  //!< Parameter that controls the amount of computation per instance pass in the fragment shader
    std::unique_ptr<opengl::VAO> array_;      //!< Proxy geometry VAO
    std::unique_ptr<opengl::VBO> vertices_;   //!< Proxy geometry VBO
    std::unique_ptr<opengl::IBO> indices_;    //!< Proxy geometry IBO
    opengl::programptr shader_;               //!< Shader program that performs the actual computation
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

