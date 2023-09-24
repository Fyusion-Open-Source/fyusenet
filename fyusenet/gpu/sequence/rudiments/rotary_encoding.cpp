//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Rotary Encoding as Positional Encoding                                      (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "rotary_encoding.h"
#include "../../rudiments/proxygenerator.h"
#include "../../../gl/shaderresource.h"
#include "../../../base/layerbase.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence::rudiments {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param width Input texture width in pixels
 * @param headDim Elements per head (head dimension)
 * @param thetaBase Base value for \f$\theta\f$
 * @param ctx GL context to operate with
 */
RotaryEncoder::RotaryEncoder(int width, int headDim, float thetaBase, const GfxContextLink& ctx) :
  GfxContextTracker(ctx) , width_(width), headDim_(headDim), thetaBase_(thetaBase) {
}


/**
 * @brief Setup required GL resources
 */
void RotaryEncoder::setup() {
    proxyGeometry();
    compileShaders();
}


/**
 * @brief Compute rotary encoding
 *
 * @param srcTexture Input texture that wraps the tensor to compute the encoding on
 * @param tokenIndex Index/offset of the start token (not applied to the texture, but to the encoding)
 * @param numTokens Number of tokens to compute the encoding for
 * @param targetRow Row offset to write the results into the target FBO
 * @param targetFBO FBO that takes the results
 *
 * @pre \c GL_SCISSOR_TEST is enabled
 */
void RotaryEncoder::forward(GLuint srcTexture, int tokenIndex, int numTokens, int targetRow, opengl::FBO *targetFBO) {
    glDisable(GL_BLEND);
    glViewport(0, targetRow, width_, numTokens);
    glScissor(0, targetRow, width_, numTokens);
    peArray_->bind();
    posEncShader_->bind();
    posEncShader_->setUniformValue("tokenIdx", tokenIndex);
    posEncShader_->setUniformVec2("viewport", width_, numTokens);
    posEncShader_->setUniformVec2("headDim", headDim_ / LayerBase::PIXEL_PACKING, headDim_);
    posEncShader_->setUniformValue("thetaBase", thetaBase_);
    targetFBO->bind();
    targetFBO->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTexture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr);
    // TODO (mw) use lines for single queries
    targetFBO->unbind();
    posEncShader_->unbind(true);
    peArray_->unbind();
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shaders to perform operation on GPU
 */
void RotaryEncoder::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    posEncShader_ = opengl::ShaderRepository::compileShaderPair("shaders/sequence/rotary_encoding.vert", "shaders/sequence/rotary_encoding.frag", preproc, typeid(this), context());
    posEncShader_->bindAttributeLocation("attributes0",0);
    posEncShader_->link();
    assert(posEncShader_->isLinked());
    if (!GLInfo::hasBinding()) {
        posEncShader_->bind();
        posEncShader_->setUniformValue("inputLayer0", 0);
        posEncShader_->unbind();
    }
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @brief Generate proxy geometry for the shader
 */
void RotaryEncoder::proxyGeometry() {
    using namespace opengl;
    const auto [arr, verts, inds] = gpu::rudiments::ProxyGenerator::simpleQuad(context());
    peArray_.reset(arr);
    peVertices_.reset(verts);
    peIndices_.reset(inds);
}


} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
