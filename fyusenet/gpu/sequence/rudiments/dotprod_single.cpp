//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Query / Key Dot-Product Computation - Single Token                          (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../../gl/shaderresource.h"
#include "../../../base/layerbase.h"
#include "../../../common/miscdefs.h"
#include "../../rudiments/proxygenerator.h"
#include "dotprod_single.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence::rudiments {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param width Full dimension (heads x head_dim) (divided by 4 and rounded up) for each token
 * @param numHeads Number of heads in the multi-head attention layer
 * @param headDim Dimension (in elements) of each head
 * @param ctx GL context to work with
 */
DotProductSingle::DotProductSingle(int width, int numHeads, int headDim, const GfxContextLink& ctx) :
  GfxContextTracker(ctx) , width_(width), numHeads_(numHeads), headDim_(headDim) {
    // TODO (mw) adjust inner batch size depending on GPU type
}


/**
 * @brief Setup GL resources for this operation
 */
void DotProductSingle::setup() {
    proxyGeometry();
    compileShaders();
}


/**
 * @brief Perform the dot-product computation
 *
 * @param queryTexture GL texture handle for the query texture
 * @param keyTexture GL texture handle for the key texture
 * @param keyLength Number of rows in the key texture
 * @param targetFBO FBO instance that wraps the target texture to write the results to
 *
 * @pre \c GL_SCISSOR_TEST is enabled
 */
void DotProductSingle::forward(GLuint queryTexture, GLuint keyTexture, int keyLength, opengl::FBO *targetFBO) {
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    int instances = (headDim_ / innerBatchSize_) / PIXEL_PACKING;
    int viewportheight = (numHeads_ + PIXEL_PACKING - 1) / PIXEL_PACKING;
    glViewport(0, 0, keyLength, viewportheight);
    glScissor(0, 0, keyLength, viewportheight);
    array_->bind();
    shader_->bind();
    shader_->setUniformVec4("inputParams", headDim_ / PIXEL_PACKING, numHeads_ / PIXEL_PACKING, keyLength, 1);
    shader_->setUniformValue("scaling", 1.0f / sqrtf((float)headDim_));
    targetFBO->bind();
    targetFBO->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, queryTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, keyTexture);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr, instances);
    targetFBO->unbind();
    shader_->unbind(true);
    array_->unbind();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shaders to perform operation on GPU
 */
void DotProductSingle::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    snprintf(preproc, sizeof(preproc) - 1, "#define INNER_BATCH_SIZE %d\n", innerBatchSize_);
    shader_ = ShaderRepository::compileShaderPair("shaders/sequence/qk_dotprod_single.vert", "shaders/sequence/qk_dotprod_single.frag", preproc, typeid(this), context());
    shader_->bindAttributeLocation("attributes0", 0);
    shader_->bindAttributeLocation("attributes1", 1);
    shader_->link();
    assert(shader_->isLinked());
    if (!GLInfo::hasBinding()) {
        shader_->bind();
        shader_->setUniformValue("inputLayer0", 0);
        shader_->setUniformValue("inputLayer1", 1);
        shader_->unbind();
    }
}


/**
 * @brief Create proxy geometry for the computation
 */
void DotProductSingle::proxyGeometry() {
    const auto [arr, verts, inds] = gpu::rudiments::ProxyGenerator::texturedQuad(context());
    array_.reset(arr);
    vertices_.reset(verts);
    indices_.reset(inds);
}


} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
