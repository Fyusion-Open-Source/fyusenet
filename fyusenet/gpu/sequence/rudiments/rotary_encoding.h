//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Rotary Encoding as Positional Encoding (Header)                             (c) Martin Wawro 2023
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

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence::rudiments {

/**
 * @brief Rotary encoding as positional encoding
 *
 * This class performs a rotary encoding operation on a sequence input where each row of the input
 * texture is treated as a sequence token. The rotary encoding is performed by rotating the input
 * elements pairwise (in 2D) following:
 *
 * \f[ \left[y_{m1} \,\,\, y_{m2} \right] =
 *      \left[ \begin{array}{cc} \cos(m\theta_1) & -\sin(m\theta_1) \\ \sin(m\theta_1) & \cos(m\theta_1) \end{array} \right]^T
 *      \left[x_{m1} \,\,\, x_{m2} \right] \f]
 *
 * where \f$ m \f$ is the token index and \f$ \theta_i \f$ is the base angle which varies across the
 * index of each token. The "base angle" \f$ \theta \f$ is a free parameter and \f$ \theta_i \f$ is
 * computed by:
 *
 * \f[ \theta_i  = \theta^{-2(i-1)/d} \f]
 *
 * where \f$ d \f$ is the dimension of a single input head.
 *
 *   The input texture is supplied in a row-by-row format, where each row represents a single token of
 *  a sequence. The width of each row corresponds to \#heads x head-dim, packed in an RGBA texture.
 *  For example for 32 heads and a head dimension of 128, a texture will look like this:
 *
 * @code
 *   32 (head_dim/4) 32 (head_dim/4)                          32 (head-dim/4)
 *  +---------------+---------------+-----------------------+------------------+
 *  |  T0(0) head0  | T0(32) head1  | ......................| T0(1023) head 32 |  m = 0
 *  +---------------+---------------+-----------------------+------------------+
 *  |  T1(0) head0  | T1(32) head1  | ......................| T1(1023) head 32 |  m = 1
 *  +---------------+---------------+-----------------------+------------------+
 *  |      ...      |      ...      |         ...           |      ...         |    ...
 *  +---------------+---------------+-----------------------+------------------+
 *  |  Tk(0) head0  | Tk(32) head1  | ......................| Tk(1023) head 32 |  m = k
 *  +---------------+---------------+-----------------------+------------------+
 * @endcode
 *
 * The output texture will be in the same format.
 */
class RotaryEncoder : public GfxContextTracker {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    RotaryEncoder(int width, int headDim, float thetaBase, const GfxContextLink& ctx);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup();
    void forward(GLuint srcTexture, int tokenIndex, int numTokens, int targetRow, opengl::FBO *targetFBO);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;
    int headDim_ = 0;
    float thetaBase_ = 0.0f;
    std::unique_ptr<opengl::VAO> peArray_;
    std::unique_ptr<opengl::VBO> peVertices_;
    std::unique_ptr<opengl::IBO> peIndices_;
    opengl::programptr posEncShader_;
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

