//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Masked SoftMax Operation on Sequences                                       (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "masked_softmax_batched.h"
#include "../../../gl/shaderresource.h"
#include "../../../gl/scoped_texturepool.h"
#include "../../gpulayerbase.h"
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
 * @param maxSeq Maximum sequence length
 * @param maxBatch Maximum batch size for a single pass
 * @param ctx GL context to work with
 */
MaskedSoftMaxBatched::MaskedSoftMaxBatched(int maxSeq, int maxBatch, const GfxContextLink& ctx) :
        GfxContextTracker(ctx) , maxSeqLen_(maxSeq), maxBatch_(maxBatch) {
}


/**
 * @brief Destructor, releases GL resources
 */
MaskedSoftMaxBatched::~MaskedSoftMaxBatched() {
    FNET_DEL_AND_CLEAR(pass1Vertices_);
    FNET_DEL_AND_CLEAR(pass2Vertices_);
    FNET_DEL_AND_CLEAR(pass2Indices_);
    FNET_DEL_AND_CLEAR(pass1Array_);
    FNET_DEL_AND_CLEAR(pass2Array_);
    FNET_DEL_AND_CLEAR(pass1FBO_);
    pass1Shader_.reset();
    pass2Shader_.reset();
    pass1Texture_.reset();
}

/**
 * @brief Setup GL resources
 *
 * @param texturePoolScope Scope ID of the texture pool to use
 *
 * This sets up internal GL resources, like proxy geometry and shaders. In addition, it also requires
 * an internal buffer texture for multi-pass rendering. This texture may be taken from the texture
 * pool if it is enabled and to avoid clashes with the owning layer, a scope has to be supplied
 * that ensures that there is no double-use of the texture.
 */
void MaskedSoftMaxBatched::setup(uint32_t texturePoolScope) {
    proxyGeometry();
    compileShaders();
    pass1Texture_ = opengl::Texture2D(1, maxSeqLen_, opengl::Texture::FLOAT32, 4, context().texturePool(), texturePoolScope, false);
    pass1FBO_ = new opengl::FBO(context(), pass1Texture_);
    if (context().texturePool()) context().texturePool()->unlockTexture(pass1Texture_);
}


/**
 * @brief Compute softmax
 *
 * @param srcTexture GL handle for source texture (input)
 * @param tokenIndex Index of the token to compute softmax for (used for masking
 * @param numTokens Number of query / tokens to process
 * @param keyLength Number of tokens in the key buffer
 * @param batchSize Size of a single batch (see long description)
 * @param targetFBO Pointer to target FBO to render to
 *
 * This function computes the softmax in a 2-pass process by first computing the denominators and
 * then use those to establish the softmax (masking is done implicitly in both cases). As this part
 * is run inside a batch-loop, the \p batchSize controls how many 4-head batches are computed
 * simultaneously. Batches will be vertically stacked in the texture layout.
 *
 * @pre \c GL_SCISSOR_TEST is enabled
 */
void MaskedSoftMaxBatched::forward(GLuint srcTexture, int tokenIndex, int numTokens, int keyLength, int batchSize, opengl::FBO *targetFBO) {
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    int numinstances = 1 + keyLength / innerBatchSize_;
    int vpheight = numTokens * batchSize;
    glViewport(0, 0, 1, vpheight);
    glScissor(0, 0, 1, vpheight);
    // ---------------------------------------------------------------
    // Pass 1: compute denominator with implied masking..
    // ---------------------------------------------------------------
    glLineWidth(1.0f);
    pass1Array_->bind();
    pass1Shader_->bind();
    pass1Shader_->setUniformVec2("viewport", 1.0f, (float)vpheight);
    pass1Shader_->setUniformVec2("inputParams", keyLength, numTokens);
    pass1Shader_->setUniformValue("baseTokenIdx", tokenIndex);
    pass1FBO_->bind();
    pass1FBO_->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTexture);
    glDrawArraysInstanced(GL_LINES, 0, batchSize*2, numinstances);
    pass1FBO_->unbind();
    pass1Shader_->unbind(true);
    pass1Array_->unbind();
    // ---------------------------------------------------------------
    // Pass 2: compute (masked) softmax...
    // ---------------------------------------------------------------
    glDisable(GL_BLEND);
    int vpwidth = keyLength;
    glViewport(0, 0, vpwidth, vpheight);
    glScissor(0, 0, vpwidth, vpheight);
    targetFBO->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    pass2Array_->bind();
    pass2Shader_->bind();
    pass2Shader_->setUniformVec4("viewport", (float) vpwidth, (float) vpheight, 1.0f, (float)maxBatch_ / (float)batchSize);
    pass2Shader_->setUniformVec2("inputParams", keyLength, numTokens);
    pass2Shader_->setUniformValue("baseTokenIdx", (int)tokenIndex, true);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pass1FBO_->getAttachment());
    glDrawElements(GL_TRIANGLES, batchSize * 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr);
    pass2Shader_->unbind();
    pass2Array_->unbind();
    targetFBO->unbind();
    glEnable(GL_BLEND);
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shaders to perform operation on GPU
 */
void MaskedSoftMaxBatched::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    float fmax = std::numeric_limits<float>::max() - 1.0f;
    snprintf(preproc, sizeof(preproc) - 1, "#define FLT_MAX %.10e\n#define INNER_BATCH_SIZE %d\n", fmax, innerBatchSize_);
    pass1Shader_ = ShaderRepository::compileShaderPair("shaders/sequence/masked_softmax_headbatch_pass1.vert", "shaders/sequence/masked_softmax_headbatch_pass1.frag", preproc, typeid(this), context());
    pass1Shader_->bindAttributeLocation("attributes0", 0);
    pass1Shader_->link();
    assert(pass1Shader_->isLinked());
    if (!GLInfo::hasBinding()) {
        pass1Shader_->bind();
        pass1Shader_->setUniformValue("inputLayer0", 0);
        pass1Shader_->unbind();
    }
    pass2Shader_ = ShaderRepository::compileShaderPair("shaders/sequence/masked_softmax_headbatch_pass2.vert", "shaders/sequence/masked_softmax_headbatch_pass2.frag", nullptr, typeid(this), context());
    pass2Shader_->bindAttributeLocation("attributes0", 0);
    pass2Shader_->bindAttributeLocation("attributes1", 1);
    pass2Shader_->link();
    assert(pass2Shader_->isLinked());
    if (!GLInfo::hasBinding()) {
        pass2Shader_->bind();
        pass2Shader_->setUniformValue("inputLayer0", 0);
        pass2Shader_->setUniformValue("inputLayer1", 1);
        pass2Shader_->unbind();
    }
}

/**
 * @brief Create proxy geometry for the shader passes
 */
void MaskedSoftMaxBatched::proxyGeometry() {
    using namespace opengl;
    // ------------------------------------------------
    // Part 1: use vertical lines for the denominator..
    // ------------------------------------------------
    auto * p1vertices = new uint32_t[1 * 2 * maxBatch_];
    pass1Array_ = new VAO(context());
    pass1Array_->bind();
    for (int i=0,offset=0; i < maxBatch_; i++) {
        p1vertices[offset++] = i;                     // top batch y-coordinate
        p1vertices[offset++] = (1<<16) | (i+1);       // bottom batch y-coordinate
    }
    pass1Vertices_ = new VBO(context());
    pass1Array_->enableArray(0);
    pass1Vertices_->setBufferData(p1vertices, 1 * 2 * maxBatch_ * sizeof(uint32_t), GL_STATIC_DRAW);
    pass1Vertices_->bind();
    pass1Array_->setVertexAttributeBuffer(0, 1, GL_UNSIGNED_INT, 0, 0);
    delete [] p1vertices;
    // ------------------------------------------------
    // Part 2: use simple quads for the final softmax
    // ------------------------------------------------
    auto * p2vertices = new float[3 * 4 * maxBatch_];
    pass2Array_ = new VAO(context());
    pass2Array_->bind();
    for (int i=0,offset=0; i < maxBatch_; i++) {
        float top = (float)i;
        float bottom = (float)(i+1);
        p2vertices[offset++] = 0.0f;
        p2vertices[offset++] = top / (float)maxBatch_;
        p2vertices[offset++] = 0.0f;

        p2vertices[offset++] = 1.0f;
        p2vertices[offset++] = top / (float)maxBatch_;
        p2vertices[offset++] = 0.0f;

        p2vertices[offset++] = 1.0f;
        p2vertices[offset++] = bottom / (float)maxBatch_;
        p2vertices[offset++] = 1.0f;

        p2vertices[offset++] = 0.0f;
        p2vertices[offset++] = bottom / (float)maxBatch_;
        p2vertices[offset++] = 1.0f;
    }
    pass2Vertices_ = new VBO(context());
    pass2Array_->enableArray(0);
    pass2Vertices_->setBufferData(p2vertices, 3 * 4 * maxBatch_ * sizeof(float), GL_STATIC_DRAW);
    pass2Vertices_->bind();
    pass2Array_->setVertexAttributeBuffer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    delete [] p2vertices;
    auto * indices = new GLshort[maxBatch_ * 6];
    pass2Indices_ = new IBO(context());
    for (int i = 0,ioffset=0; i < maxBatch_; i++) {
        int offset = i * 4;
        indices[ioffset++] = (GLshort)(offset + 0);
        indices[ioffset++] = (GLshort)(offset + 1);
        indices[ioffset++] = (GLshort)(offset + 2);
        indices[ioffset++] = (GLshort)(offset + 0);
        indices[ioffset++] = (GLshort)(offset + 2);
        indices[ioffset++] = (GLshort)(offset + 3);
    }
    pass2Indices_->setBufferData(indices, 6 * maxBatch_ * sizeof(GLshort), GL_STATIC_DRAW);
    pass2Indices_->bind();
    delete [] indices;
}


} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
