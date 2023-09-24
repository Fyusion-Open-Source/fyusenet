//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Attention DP / Value Multiplication - Single Tokens                         (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "attmul_single.h"
#include "../../../gl/shaderresource.h"
#include "../../../gl/glinfo.h"
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
 * @param width Width of the token embedding (in pixels)
 * @param numHeads Number of attention heads
 * @param headDim Dimensionality (in atoms, not pixels) of each head
 * @param ctx OpenGL context to use
 */
AttentionMulSingle::AttentionMulSingle(int width, int numHeads, int headDim, const GfxContextLink& ctx) :
  GfxContextTracker(ctx) , width_(width), numHeads_(numHeads), headDim_(headDim) {
    maxSingleWeights_ = opengl::GLInfo::getMaxVaryingVectors() - USED_VARYINGS;
}


/**
 * @brief Destructor
 */
AttentionMulSingle::~AttentionMulSingle() {
    FNET_DEL_AND_CLEAR(vertices_);
    FNET_DEL_AND_CLEAR(array_);
    shader_.reset();
}


/**
 * @brief Generate proxy geometry and setup shaders
 */
void AttentionMulSingle::setup() {
    proxyGeometry();
    compileShaders();
}


/**
 * @brief Run attention weight and value multiplication
 *
 * @param valueTexture GL texture ID of values
 * @param smTexture GL texture ID of "softmaxed" attention weights
 * @param tokenIndex Index of the (single) token in the sequence
 * @param keyLength Number of tokens stored in the key matrix
 * @param targetFBO FBO object to write the result to
 *
 * This runs the attention weight and value multiplication for a single token, writing the output
 * to the supplied \p targetFBO in a single call.
 */
void AttentionMulSingle::forward(GLuint valueTexture, GLuint smTexture, int tokenIndex, int keyLength, opengl::FBO *targetFBO) {
    assert(keyLength > 0);
    int maxweights = maxSingleWeights_ * PIXEL_PACKING;
    int instances = (tokenIndex+1 + maxweights - 1) / maxweights;
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    glLineWidth(1.0f);
    glViewport(0, 0, width_, 1);
    glScissor(0, 0, width_, 1);
    array_->bind();
    shader_->bind();
    shader_->setUniformVec2("viewport", width_, 1);
    shader_->setUniformValue("tokenIdx", tokenIndex);
    targetFBO->bind();
    targetFBO->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, valueTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, smTexture);
    glDrawArraysInstanced(GL_LINES, 0, numHeads_ * 2, instances);
    targetFBO->unbind();
    shader_->unbind();
    array_->unbind();
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile GLSL shaders to perform operation on GPU
 */
void AttentionMulSingle::compileShaders() {
    using namespace opengl;
    char preproc[256] = {0};
    snprintf(preproc, sizeof(preproc) - 1, "#define MATRIX_WEIGHTS %d\n", maxSingleWeights_);
    shader_ = ShaderRepository::compileShaderPair("shaders/sequence/att_matmul_single_masked.vert", "shaders/sequence/att_matmul_single_masked.frag", preproc, typeid(this), context());
    shader_->bindAttributeLocation("attributes0", 0);
    shader_->link();
    assert(shader_->isLinked());
    shader_->bind();
    shader_->setUniformValue("inputLayer0", 0);
    shader_->setUniformValue("attWeights", 1);
    shader_->unbind();
}


/**
 *
 *
 * Generates proxy geometry consisting of horizontal line segments for each head. Each endpoint is
 * defined as 2D vector, with the x coordinate being the position of the head and the y coordinate
 * being the head index.
 */
void AttentionMulSingle::proxyGeometry() {
    using namespace opengl;
    float * lineverts = new float[numHeads_ * 2 * 2];
    float headpos = -1.f;
    float headadd = 2.0f / (float)numHeads_;
    for (int head=0; head < numHeads_; head++) {
        lineverts[head * 4 + 0] = headpos;
        lineverts[head * 4 + 1] = (float)head;
        lineverts[head * 4 + 2] = headpos + headadd;
        lineverts[head * 4 + 3] = (float)head;
        headpos += headadd;
    }
    array_ = new VAO(context());
    array_->bind();
    vertices_ = new VBO(context());
    array_->enableArray(0);
    vertices_->setBufferData(lineverts, (GLsizei)(numHeads_ * 4 * sizeof(float)), GL_STATIC_DRAW);
    vertices_->bind();
    array_->setVertexAttributeBuffer(0, 2, GL_FLOAT, false, 0, 0);
    array_->unbind();
    delete [] lineverts;
}


} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:
