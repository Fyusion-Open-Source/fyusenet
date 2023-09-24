//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Attention Weights / Value Multiplication - Multi Tokens                     (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../../gl/shaderresource.h"
#include "../../../base/layerbase.h"
#include "../../../common/miscdefs.h"
#include "attmul_batched.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence::rudiments {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param numHeads Number of attention heads
 * @param headDim Dimensionality (in atoms, not pixels) of each head
 * @param maxSeq Maximum number of tokens that can be processed
 * @param ctx OpenGL context to use
 */
AttentionMulBatched::AttentionMulBatched(int numHeads, int headDim, int maxSeq, const GfxContextLink &ctx) :
        GfxContextTracker(ctx), numHeads_(numHeads), headDim_(headDim), maxSequenceLength_(maxSeq) {
    maxBatchedWeights_ = opengl::GLInfo::getMaxVaryingVectors() - USED_VARYINGS;
}


/**
 * @brief Destructor
 */
AttentionMulBatched::~AttentionMulBatched() {
    FNET_DEL_AND_CLEAR(vertices_);
    FNET_DEL_AND_CLEAR(array_);
    shader_.reset();
}

/**
 * @brief Generate proxy geometry and setup shaders
 */
void AttentionMulBatched::setup() {
    proxyGeometry();
    compileShaders();
}


/**
 * @brief Run attention weight and value multiplication
 *
 * @param valueTexture GL texture ID of values
 * @param smTexture GL texture ID of "softmaxed" attention weights
 * @param numTokens Number of tokens (query length) to process
 * @param tokenIndex Index of the token
 * @param headOffset Offset of the first head to process
 * @param batchSize Size of the batch to process
 * @param targetFBO FBO object to write the result to
 *
 * This runs the attention weight and value multiplication for a batch of heads, starting at the
 * provided \p headOffset. The minimum batch size is \c PIXEL_PACKING (4) as we use 4 heads per
 * pixel in parallel. Offsets as well as sizes must therefore be a multiple of 4.
 */
void AttentionMulBatched::forward(GLuint valueTexture, GLuint smTexture, int numTokens, int tokenIndex, int headOffset, int batchSize, opengl::FBO *targetFBO) {
    int fullwidth = (headDim_ * numHeads_) / LayerBase::PIXEL_PACKING;
    int vpwidth = PIXEL_PACKING * (fullwidth / numHeads_);      // we always process 4 heads at once
    array_->bind();
    shader_->bind();
    shader_->setUniformVec2("viewport", (int)vpwidth, numTokens);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    glLineWidth(1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, valueTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, smTexture);
    targetFBO->bind();
    for (int batch=0; batch < batchSize; batch++) {
        int vpxoffset = (fullwidth / numHeads_) * headOffset;
        glViewport(vpxoffset, 0, vpwidth, numTokens);
        glScissor(vpxoffset, 0, vpwidth, numTokens);
        glClear(GL_COLOR_BUFFER_BIT);
        int offset = (tokenIndex == 0) ? 0 : lines_.at(tokenIndex-1);
        int nlines = (tokenIndex == 0) ? lines_.at(numTokens - 1) : lines_.at(tokenIndex + numTokens - 1) - lines_.at(tokenIndex - 1);
        shader_->setUniformVec4("tileParams", (int)vpxoffset, (int)headDim_, batch * numTokens, tokenIndex);
        glDrawArrays(GL_LINES, offset * 2, nlines * 2);
        headOffset += PIXEL_PACKING;
    }
    targetFBO->unbind();
    shader_->unbind();
    array_->unbind();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile shaders for attention weight and value multiplication
 */
void AttentionMulBatched::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    snprintf(preproc, sizeof(preproc) - 1, "#define MATRIX_WEIGHTS %d\n", maxBatchedWeights_);
    shader_ = ShaderRepository::compileShaderPair("shaders/sequence/att_matmul_headbatch_masked.vert", "shaders/sequence/att_matmul_headbatch_masked.frag", preproc, typeid(this), context());
    shader_->bindAttributeLocation("attributes0", 0);
    shader_->link();
    assert(shader_->isLinked());
    if (!GLInfo::hasBinding()) {
        shader_->bind();
        shader_->setUniformValue("inputLayer0", 0);
        shader_->setUniformValue("attWeights", 1);
        shader_->unbind();
    }
}



/**
 * @brief Compute proxy geometry for the attention weight and value multiplication
 *
 * This function creates a vertex attribute array that is used to render the attention weight and
 * value multiplication. The proxy geometry here consists of horizontal lines, which are duplicated
 * according to a triangular arrangement of the masked attention weights (we assume that we always
 * deal with a full causal mask for batched processing).
 *
 * The number of lines is based on the maximum number tokens and on the size of the interface
 * storage between the vertex- and the fragment shader, as the vertex shader buffers the coefficients
 * from the attention weights to minimize texture lookups.
 */
void AttentionMulBatched::proxyGeometry() {
    using namespace opengl;
    int totallines = 0;
    for (int query=0; query < maxSequenceLength_; query++) totallines += (query + maxBatchedWeights_) / maxBatchedWeights_;
    GLuint * lineverts = new GLuint[totallines * 2];
    lines_.reserve(totallines);
    for (int query=0,offset=0; query < maxSequenceLength_; query++) {
        int dups = (query + maxBatchedWeights_) / maxBatchedWeights_;
        for (int dup=0; dup < dups; dup++) {
            assert(dup < (1<<15));
            lineverts[offset++] = (query<<16) | (dup<<1);
            lineverts[offset++] = (query<<16) | (dup<<1)|1;
        }
        lines_.emplace_back(offset / 2);
    }
    array_ = new VAO(context());
    array_->bind();
    vertices_ = new VBO(context());
    array_->enableArray(0);
    vertices_->setBufferData(lineverts, (GLsizei)(totallines * 2 * sizeof(GLuint)), GL_STATIC_DRAW);
    vertices_->bind();
    array_->setVertexAttributeBuffer(0, 1, GL_UNSIGNED_INT, 0, 0);
    array_->unbind();
    delete [] lineverts;
}


} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
