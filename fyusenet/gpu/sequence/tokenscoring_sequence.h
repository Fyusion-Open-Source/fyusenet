//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Token-Scoring Layer for Sequences (Header)                                  (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/ubo.h"
#include "../../gl/texture.h"
#include "../../base/bufferspec.h"
#include "../gpulayerbase.h"
#include "../tokenscoringlayerbuilder.h"

class SequenceLayerTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence {

class EmbeddingLayer;

/**
 * @brief Layer that performs token scoring and selection for generative sequence learning
 *
 * This layer takes a set of embeddings (one per row) and computes the inner product (similarity)
 * between the last row of the embeddings and an internal vocabulary. The inner products are
 * then ranked and a result is selected and written into a supplied output texture (at the first
 * row). The output texture can then be used as input for an autoregressive sequence generator.
 *
 * In addition, this layer also supports the (asynchronous) download of the predicted token for
 * the control al
 *
 */
class TokenScoringLayer : public gpu::GPULayerBase {
    friend class ::SequenceLayerTest;
 public:
    constexpr static int HARD_TOKEN_TEXTURE_MAX = 8;
    constexpr static int SCATTER_WIDTH = 128;
    constexpr static int MAX_VOCAB_AGGREGATE_SIZE = 64;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit TokenScoringLayer(const TokenScoringLayerBuilder & builder);
    TokenScoringLayer(const TokenScoringLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * token) override;
    void setup() override;
    void cleanup() override;
    void setupFBOs() override;
    void updateFBOs() override;
    virtual void cloneEmbeddingTable(const EmbeddingLayer& src);
    void loadParameters(const ParameterProvider * source) override;
    [[nodiscard]] GPUBuffer *getGPUOutputBuffer(int port) const override;
    [[nodiscard]] GPUBuffer *getGPUInputBuffer(int port) const override;

    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    void writeResult(const char *fileName, bool includePadding) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;
    [[nodiscard]] BufferSpec::dtype getInputType(int port) const override;
    [[nodiscard]] BufferSpec::dtype getOutputType(int port) const override;
    void proxyGeometry();
    void compileShaders();
    void prepShaders();
    void projectToken(int token);
    void setupProjectionTexture();
    void flatten();
    void scatter();
    void selection();
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int embedDim_ = 0;                            //!< Embedding dimension (number of channels)
    int texWidth_ = 0;                            //!< Width of embedding texture array (in pixels)
    int tableRows_ = 0;                           //!< (Full) height of the supplied embedding table
    ScoringType scoring_;                         //!< Type of scoring to be used in this layer, needs to be compatible with the subsequent selection layer
    float temperature_ = 0.f;                     //!<
    int topK_ = 0;                                //!<
    float topP_ = 1.f;                            //!<
    GLuint scatterDepth_ = 0;                     //!< Renderbuffer ID for the scatter pass
    int projectionSize_[2] = {0};                 //!<
    int flatSubsampling_[2] = {0};
    int proInstanceWidth_ = 8;                    //!< Width of segmented dot-product computation for the projection part per instance
    int vocabAggregateSize_ = 0;                  //!< Number of vocabulary items to be aggregated in the first pass of the flattening
    std::vector<int> projectionSegments_;         //!< Vertical segment sizes for the projection part
    opengl::VAO * proArray_ = nullptr;            //!<
    opengl::VBO * proVerts_ = nullptr;            //!<
    opengl::IBO * proIndices_ = nullptr;          //!<
    opengl::VAO * pass1FlatArray_ = nullptr;      //!<
    opengl::VBO * pass1FlatVerts_ = nullptr;      //!<
    opengl::VAO * scatterArray_ = nullptr;        //!<
    opengl::VBO * scatterVerts_ = nullptr;        //!<
    opengl::programptr pass1FlatShader_;          //!<
    opengl::programptr pass2FlatShader_;          //!<
    opengl::programptr scatterShader_;            //!<
    opengl::programptr selectionShader_;          //!<
    opengl::Texture2D projectionTexture_;         //!<
    opengl::Texture2D scatterMatches_;            //!<
    opengl::FBO * projectionFBO_ = nullptr;       //!< FBO to be used to render into the #projectionTexture_ to determine scoring for all items in the vocabulary
    opengl::FBO * flatFBOs_[2] = {nullptr};       //!< FBOs to be used for rendering a single pixel that (hopefully) contains a bit of the image statistics
    opengl::FBO * scatterFBO_ = nullptr;          //!<
    opengl::FBO * selectionFBO_ = nullptr;        //!< Internal FBO for the selection pass (used for async downloading)

    /**
     * @brief Array of textures containing the embedding table
     */
    std::vector<opengl::Texture2D> embeddingTextures_;
    /**
     *
     */
    opengl::programptr proShader_;
};

} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
