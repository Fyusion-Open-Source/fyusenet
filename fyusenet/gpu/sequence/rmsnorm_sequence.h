//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Root Mean Square Norm for Sequences (Header)                                (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpulayerbase.h"
#include "../gpulayerbuilder.h"

class AttentionTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence {

/**
 * @brief Root-mean-square norm for sequences
 *
 * This class implements root-mean-square normalization for sequence tensors. The norm is
 * computed for on a token-by-token basis by using the following normalizer for each token
 * \f$ t_i \f$:
 * \f[  n_i = \frac{\sqrt{\sum_j t_{ij}^2}}{|t_i|} \f]
 * where \f$ j \f$ subscripts along the embedding dimension of each token.
 *
 * And then replacing \f$ t_i \f$ by \f$ \frac{t_i}{n_i} \f$.
 *
 * The input data is assumed to be in the following format:
 * @code
 *  +--------------------------------------------------------------------+
 *  |  e0  |  e1  | e2  |  e3  |  e4  |  e5  |  e6  |     ...   |  e<d>  | token 0
 *  +--------------------------------------------------------------------+
 *  |  e0  |  e1  | e2  |  e3  |  e4  |  e5  |  e6  |     ...   |  e<d>  | token 1
 *  +--------------------------------------------------------------------+
 *  |                          .............                             | token ...
 *  +--------------------------------------------------------------------+
 *  |  e0  |  e1  | e2  |  e3  |  e4  |  e5  |  e6  |     ...   |  e<d>  | token N
 *  +--------------------------------------------------------------------+
 * @endcode
 *
 * where each row in the texture represents a token and each pixel in a row represents 4 consecutive
 * entries in each embedding vector (RGBA-format texture).
 *
 * The output texture format will be identical to the input format. Depending on the number of
 * rows of the input matrix, either a \e short shader will be used or a \e long shader. The
 * short shader is meant for single row inputs only, as they often occur during the autoregressive
 * prediction of single (output) tokens. For a single token, this layer only requires one shader
 * pass.
 *
 * The long shader is meant for inputs with multiple rows and employs a two-pass approach which
 * first computes the norm for each row and then performs the actual normalization in a 2nd
 * pass.
 */
class RMSNormLayer : public gpu::GPULayerBase {
    friend class ::AttentionTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    RMSNormLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void setup() override;
    void cleanup() override;
    void setupFBOs() override;
    void updateFBOs() override;
    void loadParameters(const ParameterProvider * provider) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] GPUBuffer *getGPUOutputBuffer(int port) const override;
    [[nodiscard]] GPUBuffer *getGPUInputBuffer(int port) const override;
    void writeResult(const char *fileName, bool includePadding) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;
    void computeShortSequence();
    void computeLongSequence();
    void proxyGeometry();
    void compileShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int embedDim_ = 0;                      //!< Embedding dimension of the input tensor (width, not necessarily the texture width)
    int sequenceLength_ = 0;                //!< Number of rows of the input tensor (not necessarily the texture height)
    int contraction_ = 0;                   //!< Number of items to contract per instance for the 1st pass norm computation
    int instances_ = 0;                     //!< Number of instances to use for the 1st pass norm computation
    opengl::Texture2D normTex_;             //!< Backing texture for long shader 1st pass (norm computation)
    VAO * pass1ArrayLong_ = nullptr;        //!< VAO for 1st pass of long shader (and also the only pass of the short shader)
    VBO * pass1VerticesLong_ = nullptr;     //!< VBO for 1st pass of long shader
    VAO * pass2ArrayLong_ = nullptr;        //!< VAO for 2nd pass of long shader
    VBO * pass2VerticesLong_ = nullptr;     //!< VBO for 2nd pass of long shader
    IBO * quadIndices_ = nullptr;           //!< IBO for quad geometry/indices
    programptr pass1ShaderLong_;            //!< Shader program for 1st pass of long shader
    programptr pass2ShaderLong_;            //!< Shader program for 2nd pass of long shader
    programptr shortShader_;                //!< Shader program for short shader
    FBO * normFBO_ = nullptr;               //!< Pointer to %FBO that stores the norm
    GLuint weightTexture_ = 0;              //!< GL texture ID for the weights used by this layer
};

} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
