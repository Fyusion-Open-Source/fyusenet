//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Query / Key Dot-Product Computation - Multi Tokens                          (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "dotprod_batched.h"
#include "../../../gl/shaderresource.h"
#include "../../../base/layerbase.h"
#include "../../../common/miscdefs.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence::rudiments {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param numHeads Number of heads in the multi-head attention layer
 * @param maxBatch Maximum allowed batch size
 * @param headDim Dimension (in elements) of each head
 * @param ctx GL context to work with
 */
DotProductBatched::DotProductBatched(int numHeads, int headDim, int maxBatch, const GfxContextLink &ctx) :
        GfxContextTracker(ctx), numHeads_(numHeads), headDim_(headDim), maxBatch_(maxBatch) {
}


/**
 * @brief Destructor, releases GL resources
 */
DotProductBatched::~DotProductBatched() {
    FNET_DEL_AND_CLEAR(array_);
    FNET_DEL_AND_CLEAR(vertices_);
    FNET_DEL_AND_CLEAR(indices_);
    shader_.reset();
}

/**
 * @brief Setup GL resources for this operation (proxy geometry and shaders)
 */
void DotProductBatched::setup() {
    proxyGeometry();
    compileShaders();
}

/**
  * @brief Perform the dot-product computation
*
 * @param queryTexture GL texture handle for the query texture
 * @param keyTexture GL texture handle for the key texture
 * @param numTokens
 * @param keyLength Number of rows in the key texture
 * @param headOffset
 * @param batchSize
 * @param targetFBO FBO instance that wraps the target texture to write the results to
 *
 * Runs the dot-product computation for the given query and key textures. Depending on the
 * \p batchSize, will render a set of tiles to the output texture.
 * In order to complete the full dot product, multiple instances of those tiles will be rendered and
 * composed using the ROPs.
 *
 * @pre \c GL_SCISSOR_TEST test is enabled
 */
void DotProductBatched::forward(GLuint queryTexture, GLuint keyTexture, int numTokens, int keyLength, int headOffset, int batchSize, opengl::FBO *targetFBO) {
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    int numinstances = (headDim_ / PIXEL_PACKING) / innerBatchSize_;
    int viewportwidth = keyLength;
    int viewportheight = numTokens * batchSize;
    glViewport(0, 0, viewportwidth, viewportheight);
    glScissor(0, 0, viewportwidth, viewportheight);
    array_->bind();
    shader_->bind();
    shader_->setUniformVec4("viewport", (float)viewportwidth, (float)viewportheight, 1.0f, (float)maxBatch_ / (float)batchSize);
    shader_->setUniformVec4("sizeParams", headDim_ / PIXEL_PACKING, numHeads_, keyLength, numTokens);
    shader_->setUniformValue("headOffset", headOffset);
    shader_->setUniformValue("scaling", 1.0f / sqrtf((float)headDim_));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, queryTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, keyTexture);
    targetFBO->bind();
    targetFBO->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElementsInstanced(GL_TRIANGLES, batchSize * 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr, numinstances);
    targetFBO->unbind();
    shader_->unbind();
    array_->unbind();
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shader to perform operation on GPU
 */
void DotProductBatched::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    snprintf(preproc, sizeof(preproc) - 1, "#define INNER_BATCH_SIZE %d\n", innerBatchSize_);
    shader_ = ShaderRepository::compileShaderPair("shaders/sequence/qk_dotprod_headbatch.vert", "shaders/sequence/qk_dotprod_headbatch.frag", preproc, typeid(this), context());
    shader_->bindAttributeLocation("attributes0", 0);
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
 *
 * This creates an arrangement of tiles (quads) that cover the target texture in horizontal stripes.
 * Depending on the batch size, a different number of tiles is selected for rendering.
 */
void DotProductBatched::proxyGeometry() {
    using namespace opengl;
     //-----------------------------------------------------
     // Setup proxy geometry for larger scale batches...
     //-----------------------------------------------------
     array_ = new VAO(context());
     array_->bind();
     float *verts = new float[maxBatch_ * 4 * 4];
     for (int tile=0, offset=0; tile < maxBatch_; tile++) {
         float top = ((float)tile / (float)maxBatch_) * 2.0f - 1.0f;
         float bottom = ((float)(tile+1) / (float)maxBatch_) * 2.0f - 1.0f;
         verts[offset++] = -1.0f;
         verts[offset++] = top;
         verts[offset++] = (float)(tile * 4);
         verts[offset++] = 0.0f;
         verts[offset++] = 1.0f;
         verts[offset++] = top;
         verts[offset++] = (float)(tile * 4);
         verts[offset++] = 0.0f;
         verts[offset++] = 1.0f;
         verts[offset++] = bottom;
         verts[offset++] = (float)(tile * 4);
         verts[offset++] = 1.0f;
         verts[offset++] = -1.0f;
         verts[offset++] = bottom;
         verts[offset++] = (float)(tile * 4);
         verts[offset++] = 1.0f;
     }
     vertices_ = new VBO(context());
     array_->enableArray(0);
     vertices_->setBufferData(verts, (GLsizei)(maxBatch_ * 4 * 4 * sizeof(float)), GL_STATIC_DRAW);
     vertices_->bind();
     array_->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
     delete [] verts;
     //---------------------------------------------
     // IBO part
     //---------------------------------------------
     auto *indices = new GLshort[maxBatch_ * 6];
     indices_ = new IBO(context());
     for (int i = 0; i < maxBatch_; i++) {
         int offset = i * 4;
         indices[i * 6 + 0] = (GLshort)(offset + 0);
         indices[i * 6 + 1] = (GLshort)(offset + 1);
         indices[i * 6 + 2] = (GLshort)(offset + 2);
         indices[i * 6 + 3] = (GLshort)(offset + 0);
         indices[i * 6 + 4] = (GLshort)(offset + 2);
         indices[i * 6 + 5] = (GLshort)(offset + 3);
     }
     indices_->setBufferData(indices, (GLsizei)(6 * maxBatch_ * sizeof(GLshort)), GL_STATIC_DRAW);
     indices_->bind();
     delete [] indices;
     array_->unbind();
     indices_->unbind();
     vertices_->unbind();
}


} // fyusion::fyusenet::gpu::seequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
