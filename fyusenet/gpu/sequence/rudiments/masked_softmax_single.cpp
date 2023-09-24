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

#include "../../../gl/scoped_texturepool.h"
#include "../../../gl/shaderresource.h"
#include "../../../base/layerbase.h"
#include "../../../common/miscdefs.h"
#include "masked_softmax_single.h"

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
 * @param headDim Dimension (in elements) of each head
 * @param ctx GL context to work with
 */
MaskedSoftMaxSingle::MaskedSoftMaxSingle(int numHeads, int headDim, const GfxContextLink& ctx) :
  GfxContextTracker(ctx) , numHeads_(numHeads), headDim_(headDim) {
}


/**
 * @brief Destructor, releases GL resources
 */
MaskedSoftMaxSingle::~MaskedSoftMaxSingle() {
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
 * @brief Setup GL resources for this operation
 *
 * @param texturePoolScope Scope for the texture pool to use (if enabled)
 *
 * This sets up internal GL resources, like proxy geometry and shaders. In addition, it also requires
 * an internal buffer texture for multi-pass rendering. This texture may be taken from the texture
 * pool if it is enabled and to avoid clashes with the owning layer, a scope has to be supplied
 * that ensures that there is no double-use of the texture.
 */
void MaskedSoftMaxSingle::setup(uint32_t texturePoolScope) {
    proxyGeometry();
    compileShaders();
    // make sure we get a fresh texture as we do not use interface textures here
    pass1Texture_ = opengl::Texture2D(1, (numHeads_ + PIXEL_PACKING-1) / PIXEL_PACKING, opengl::Texture::FLOAT32, 4, context().texturePool(), texturePoolScope, false);
    pass1FBO_ = new opengl::FBO(context(), pass1Texture_);
    if (context().texturePool()) context().texturePool()->unlockTexture(pass1Texture_);
}


/**
 * @brief Compute softmax
 *
 * @param srcTexture GL handle for source texture (input)
 * @param tokenIndex Index of the token to compute softmax for (used for masking
 * @param keyLength Number of tokens in the key buffer
 * @param targetFBO Pointer to target FBO to render to
 *
 * @pre \c GL_SCISSOR_TEST is enabled
 */
void MaskedSoftMaxSingle::forward(GLuint srcTexture, int tokenIndex, int keyLength, opengl::FBO *targetFBO) {
    int instances = 1 + tokenIndex / innerBatchSize_;
    int vpheight = (numHeads_ + LayerBase::PIXEL_PACKING - 1) / PIXEL_PACKING;
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    glLineWidth(1.0f);
    // ------------------------------------------------
    // Pass 1: compute (masked) denominators for the
    //         softmax computation...
    // ------------------------------------------------
    glViewport(0, 0, 1, vpheight);
    glScissor(0, 0, 1, vpheight);
    pass1Array_->bind();
    pass1Shader_->bind();
    pass1Shader_->setUniformVec2("viewport", 1.0f, (float)vpheight);
    pass1Shader_->setUniformValue("tokenIdx", tokenIndex);
    pass1Shader_->setUniformValue("keyLength", keyLength);
    pass1FBO_->bind();
    pass1FBO_->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTexture);
    glDrawArraysInstanced(GL_LINES, 0, 2, instances);
    pass1FBO_->unbind();
    pass1Shader_->unbind(true);
    pass1Array_->unbind();
    // ------------------------------------------------
    // Pass 2: actual (masked) softmax computation...
    // ------------------------------------------------
    glDisable(GL_BLEND);
    glViewport(0, 0, keyLength, vpheight);
    glScissor(0, 0, keyLength, vpheight);
    pass2Array_->bind();
    pass2Shader_->bind();
    pass2Shader_->setUniformVec2("viewport", (float)keyLength, (float)vpheight);
    pass2Shader_->setUniformValue("tokenIdx", tokenIndex);
    targetFBO->bind();
    targetFBO->setWriteMask();
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pass1FBO_->getAttachment());
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr);
    targetFBO->unbind();
    pass2Shader_->unbind();
    pass2Array_->unbind();
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shaders to perform operation on GPU
 */
void MaskedSoftMaxSingle::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    float fmax = std::numeric_limits<float>::max() - 1.0f;
    snprintf(preproc, sizeof(preproc) - 1, "#define FLT_MAX %.10e\n#define INNER_BATCH_SIZE %d\n", fmax, innerBatchSize_);
    pass1Shader_ = ShaderRepository::compileShaderPair("shaders/sequence/masked_softmax_single_pass1.vert", "shaders/sequence/masked_softmax_single_pass1.frag", preproc, typeid(this), context());
    pass1Shader_->bindAttributeLocation("attributes0", 0);
    pass1Shader_->link();
    assert(pass1Shader_->isLinked());
    if (!GLInfo::hasBinding()) {
        pass1Shader_->bind();
        pass1Shader_->setUniformValue("inputLayer0", 0);
        pass1Shader_->unbind();
    }
    pass2Shader_ = ShaderRepository::compileShaderPair("shaders/sequence/masked_softmax_single_pass2.vert", "shaders/sequence/masked_softmax_single_pass2.frag", nullptr, typeid(this), context());
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
 * @brief Create proxy geometry for the computation
 */
void MaskedSoftMaxSingle::proxyGeometry() {
    using namespace opengl;
    // ------------------------------------------------
    // Part 1: use a vertical line for the denominator..
    // ------------------------------------------------
    uint32_t p1vertices[] = {0, 1};
    pass1Array_ = new VAO(context());
    pass1Array_->bind();
    pass1Vertices_ = new VBO(context());
    pass1Array_->enableArray(0);
    pass1Vertices_->setBufferData(p1vertices,  2 * sizeof(uint32_t), GL_STATIC_DRAW);
    pass1Vertices_->bind();
    pass1Array_->setVertexAttributeBuffer(0, 1, GL_UNSIGNED_INT, 0, 0);
    // ------------------------------------------------
    // Part 2: use simple quad for the final softmax
    // ------------------------------------------------
    float p2vertices[] = {-1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f, 1.0f,
                          -1.0f, 1.0f};
    pass2Array_ = new VAO(context());
    pass2Array_->bind();
    pass2Vertices_ = new VBO(context());
    pass2Array_->enableArray(0);
    pass2Vertices_->setBufferData(p2vertices, 2 * 4 * sizeof(float), GL_STATIC_DRAW);
    pass2Vertices_->bind();
    pass2Array_->setVertexAttributeBuffer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    GLshort indices[] = {0, 1, 2, 0, 2, 3};
    pass2Indices_ = new IBO(context());
    pass2Indices_->setBufferData(indices, 6 * sizeof(GLshort), GL_STATIC_DRAW);
    pass2Indices_->bind();
}


} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
