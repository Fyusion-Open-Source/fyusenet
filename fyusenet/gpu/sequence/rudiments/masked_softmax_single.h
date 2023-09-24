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
 * @brief Causally-masked SoftMax operation for the single token case
 *
 * This class computes a causally-masked softmax operator on an input texture that contains the
 * dot-product result of a multi-head attention layer.
 * The special case of only having a single query token for the attention computations is a regular
 * case due to the autoregressive way tokens are predicted and this class specialized in computing
 * it efficiently.
 *
 * It should be noted that this class does not just compute the softmax of the input, but rather
 * a causally-masked softmax that only considers the tokens on the key side that have an index
 * equal or smaller than the query index (which should be all of them in the regular case).
 *
 * \include msm_single_appendix.inc
 */
class MaskedSoftMaxSingle : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SoftMaxTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    MaskedSoftMaxSingle(int numHeads, int headDim, const GfxContextLink& ctx);
    ~MaskedSoftMaxSingle() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup(uint32_t texturePoolScope);
    void forward(GLuint srcTexture, int tokenIndex, int keyLength, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int numHeads_ = 0;                          //!< Number of heads in the attention layer
    int headDim_ = 0;                           //!< Dimension of each head
    int innerBatchSize_ = 16;                   //!< Parameter that controls the amount of computation per instance pass in the fragment shader
    opengl::VAO * pass1Array_ = nullptr;        //!< VAO for the first pass (denominator computation)
    opengl::VBO * pass1Vertices_ = nullptr;     //!< VBO for the first pass (denominator computation)
    opengl::VAO * pass2Array_ = nullptr;        //!< VAO for the second pass (softmax computation)
    opengl::VBO * pass2Vertices_ = nullptr;     //!< VBO for the second pass (softmax computation)
    opengl::IBO * pass2Indices_ = nullptr;      //!< IBO for the second pass (softmax computation)
    opengl::programptr pass1Shader_;            //!< Shader for the first pass (denominator computation)
    opengl::programptr pass2Shader_;            //!< Shader for the second pass (softmax computation)
    opengl::Texture2D pass1Texture_;            //!< Buffer/intermediary texture for denominators
    opengl::FBO * pass1FBO_ = nullptr;          //!< FBO for the first pass (denominator computation), writes to #pass1Texture_
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

