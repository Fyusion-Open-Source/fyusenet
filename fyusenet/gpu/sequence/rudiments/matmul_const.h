//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Matrix/Matrix Multiplication w/ constant Matrix (Header)                    (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <cstdint>
#include <cstring>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../../gl/gl_sys.h"
#include "../../../gl/shaderprogram.h"
#include "../../../gl/fbo.h"
#include "../../../gl/vao.h"
#include "../../../gl/vbo.h"
#include "../../../gl/ibo.h"
#include "../../../gl/texture.h"
#include "../../../base/layerbase.h"
#include "../../../base/parameterprovider.h"
#include "../../rudiments/preamblegenerator.h"

class AttentionTest;
class SequenceTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence::rudiments {

/**
 * @brief Perform matrix / matrix or matrix / vector multiplication with a constant matrix
 *
 * This class performs a multiplication of two matrices, where the right matrix is a constant
 * matrix that has been uploaded to the GPU before. The left matrix arises from chained
 * computations by the layers in the network.
 * This particular implementation runs on texture layouts used for processing of \e sequences.
 *
 * The operation that is carried out is given by:
 *
 * \f[ \mathbf{Y} = \mathbf{X}\mathbf{W} \f]
 *
 * where \f$ \mathbf{X} \in \mathbb{R}^{n \times m}\f$ is allowed to degenerate into a vector
 * \f$ \mathbf{x} \in \mathbb{R}^{1 \times m}\f$. In the latter case, this class also supports
 * to add a \e bias to the result of the multiplication to yield the affine transform:
 *
 * \f[ \mathbf{y} = \mathbf{x}\mathbf{W} + \mathbf{b} \f]
 *
 * where \f$ \mathbf{b} \in \mathbb{R}^{1 \times m} \f$.
 *
 * In addition to that, this class allows for an additional \e residual input as well as for
 * blending into an existing residual. For the additional residual, this class will accept an
 * additional texture as residual and compute:
 *
 * \f[ \mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{R}\f]
 *
 * Whereas for blending/adding to an existing residual, this class will simply blend the results
 * of the multiplication into the existing residual. Which one of those options (yielding the
 * same results) is used, depends on the underlying use-case.
 *
 * \include{doc} seq_matmul_appendix.inc
 *
 * @warning The current implementation only handles 4-bit quantized data as of now.
 */
class MatMulConst : public GfxContextTracker {
    friend class ::AttentionTest;
    friend class ::SequenceTest;
 public:

    /**
     * Shader types used to run the matrix multiplication
     */
    enum shtype: uint8_t {
        VERT_SHORT = 0,     //!< Vertex shader for short sequences
        FRAG_SHORT,         //!< Fragment shader for short sequences
        VERT_LONG,          //!< Vertex shader for long sequences
        FRAG_LONG,          //!< Fragment shader for long sequences
        ANY_SHORT,          //!< Either fragment or vertex shader for short sequences (used for custom preprocessing)
        ANY_LONG            //!< Either fragment or vertex shader for long sequences (used for custom preprocessing)
    };

    constexpr static int MMUL_WEIGHTS_PER_PASS = 8 * LayerBase::PIXEL_PACKING;  // FIXME (mw) this is valid only for 4-bit quantization

    // TODO (mw) find something dynamic / system-specific here
#ifdef HIGH_PRECISION
    constexpr static int MATMUL_LONG_THRESHOLD = 16;
#else
    constexpr static int MATMUL_LONG_THRESHOLD = 8;
#endif
    constexpr static int INPUT0_UNIT = 0;
    constexpr static int INPUT1_UNIT = 1;
    constexpr static int BIAS_UNIT = 5;
    constexpr static int RESIDUAL_UNIT = 6;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    // FIXME (mw) this is just ugly
    MatMulConst(const gpu::rudiments::PreambleGenerator& preamble, int rows, int columns, int maxSeq,
                param_type dataType, int qGroupSize,
                bool bias, bool inputResidual, bool outputResidual,
                const GfxContextLink &ctx);
    ~MatMulConst() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void customShader(shtype shaderType, const char * resource);
    void customShaderPreproc(const std::function<void(char *, size_t, shtype)> & preprocFunc);
    void customShaderPostproc(const std::function<void(opengl::ShaderProgram *, shtype)>& postFunc);
    void setup();
    void forward(int dataRows, int outputRowOffset, opengl::FBO *targetFBO);
    void loadWeights(const DataBlob & weights);
    void loadBiases(const DataBlob & weights);
    void loadQuantizationTables(const DataBlob& scales, const DataBlob& zeros);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void weightMatMulLong4Bit(opengl::FBO *target, int dataRows);
    void weightMatMulShort4Bit(opengl::FBO *target, int dataRows);
    void proxyGeometry();
    void compileShaders();
    void postProcessShader(ShaderProgram * shader, shtype type);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::unique_ptr<opengl::VAO> array_;      //!< Vertex array for proxy geometry
    std::unique_ptr<opengl::VBO> vertices_;   //!< Vertex buffer for proxy geometry
    opengl::programptr shaderShort_;          //!< Short matrix multiplication shader
    opengl::programptr shaderShortPrime_;     //!< Short matrix multiplication shader (1st pass for bias / residual)
    opengl::programptr shaderLong_;           //!< Long matrix multiplication shader
    opengl::programptr shaderLongPrime_;      //!< Long matrix multiplication shader (1st pass for bias / residual)
    int rows_ = 0;                            //!< Number of rows in the weight matrix
    int columns_ = 0;                         //!< Number of columns in the weight matrix (without any pixel packing tricks)
    int outputWidth_ = 0;                     //!< Output width (in pixels) of the resulting tensor
    int outputHeight_ = 0;                    //!< Maximum or exact number of output rows of the resulting tensor
    bool hasBias_ = false;                    //!< Set to true if this is an affine transform (i.e. adds a bias after performing multiplication)
    bool inResidual_ = false;                 //!< Indicator whether an explicit (input) residual is to be added to the output
    bool outResidual_ = false;                //!< Indicator whether the layer should treat the output texture as residual and blend do it
    bool isQuantized_ = false;                //!< Set to true if the operation uses (integer) quantized weights
    GLuint weightData_ = 0;                   //!< OpenGL texture handle for the weight matrix
    GLuint scaleData_ = 0;                    //!< OpenGL texture handle for the quantization scales
    GLuint zeroData_ = 0;                     //!< OpenGL texture handle for the quantization zeros
    GLuint biasData_ = 0;                     //!< OpenGL texture handle for the bias vector
    int quantGroupSize_ = 0;                  //!< For quantized weight matrices, defines the quantization group size
    int smallMWPacks_ = 1;                    //!< Number of internal matrix-weight packs to be used for short matrix multiplications

    /**
     * Optional override for shader preparation (post compilation / link)
     */
    std::function<void(opengl::ShaderProgram *, shtype)> customShaderPost_;

    /**
     * Optional override for shader preprocessor
     */
    std::function<void(char *, size_t, shtype)> customShaderPreproc_;

    /**
     * Optional overrides for shader resource names to tweak the operation in this class
     */
    const char * customShaders_[4] = {nullptr};

    /**
     * Number of weight-pack "lanes" (sets of matrix weights decodeable from a single quantized
     * 128-bit RGBA portion) to send from vertex to fragment shader on long multiplications.
     */
    int weightLanes_ = 1;

    /**
     * Data type for the weights supplied to the MatMul operation.
     */
    param_type dataType_ = param_type::WGT_FLOAT;

    /**
     * Generator for preambles to be handed into the shader preprocessor
     */
    gpu::rudiments::PreambleGenerator preamble_;
};

} // fyusion::fyusenet::gpu::sequence::rudiments namespace

// vim: set expandtab ts=4 sw=4:

