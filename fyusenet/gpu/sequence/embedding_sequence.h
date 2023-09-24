//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Embedding Layer for Sequences (Header)                                      (c) Martin Wawro 2023
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
#include "../gpulayerbuilder.h"
#include "../embeddinglayerbuilder.h"

class SequenceLayerTest;

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::sequence {

/**
 * @brief Embedding layer for sequences
 *
 * This layer is to be used to compute embeddings for sequences, based on (unsigned) 32-bit
 * input tokens.
 *
 */
class EmbeddingLayer : public gpu::GPULayerBase {
    friend class ::SequenceLayerTest;
    friend class TokenScoringLayer;
 public:
    constexpr static int HARD_TOKEN_TEXTURE_MAX = 8;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit EmbeddingLayer(const EmbeddingLayerBuilder & builder);
    EmbeddingLayer(const EmbeddingLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void setup() override;
    void cleanup() override;
    void setupFBOs() override;
    void updateFBOs() override;
    void loadParameters(const ParameterProvider * source) override;
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
    void compileShader();
    void prepShader(ShaderProgram * shader);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int embedDim_ = 0;                              //!< Embedding dimension (number of channels)
    int texWidth_ = 0;                              //!< Width of embedding texture(s) (in pixels)
    int tableRows_ = 0;                             //!< (Full) height of the supplied embedding table
    int sequenceLength_ = 0;                        //!< Sequence length of last query
    param_type srcType_ = param_type::WGT_DEFAULT;  //!< Data type of the embedding table on CPU
    param_type devType_ = param_type::WGT_DEFAULT;  //!< Data type of the embedding table on compute device
    opengl::VAO * array_ = nullptr;                 //!< Vertex array object for proxy geometry
    opengl::VBO * vertices_ = nullptr;              //!< Vertex buffer object for proxy geometry
    opengl::programptr shader_;                     //!< Embedding shader
    std::vector<Texture2D> embeddingTextures_;      //!< Array of embedding textures
};

} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
