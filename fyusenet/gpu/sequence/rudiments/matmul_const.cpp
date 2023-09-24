//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Matrix/Matrix Multiplication w/ constant Matrix                             (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "matmul_const.h"
#include "../../rudiments/linear_texture_loader.h"
#include "../../../gl/shaderresource.h"
#include "../../../common/miscdefs.h"
#include "../../gpulayerbase.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence::rudiments {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param preamble Preamble generator to use for generating preprocessor macro defs
 * @param rows Number of rows in the weight matrix
 * @param columns Number of columns in the weight matrix
 * @param maxSeq Maximum sequence length this multiplier shall be able to process
 * @param dataType Data type for the weights
 * @param qGroupSize Quantization group size for the weights (set to 0 if there is no quantization)
 * @param bias Whether to add bias to the matrix multiplication (i.e. make it an affine transformation)
 * @param inputResidual Whether to add an input residual to the matrix multiplication
 * @param outputResidual Whether to add the result of the matrix multiplication to an output residual via blending
 * @param ctx Context link to operate in
 */
MatMulConst::MatMulConst(const gpu::rudiments::PreambleGenerator& preamble,
                         int rows, int columns, int maxSeq,
                         param_type dataType, int qGroupSize,
                         bool bias, bool inputResidual, bool outputResidual, const GfxContextLink &ctx) :
        GfxContextTracker(ctx), rows_(rows), columns_(columns), outputHeight_(maxSeq),
        hasBias_(bias), inResidual_(inputResidual), outResidual_(outputResidual), quantGroupSize_(qGroupSize), dataType_(dataType) ,
        preamble_(preamble) {

    // FIXME (mw) make sure that quantgroupsize is within operational bounds

    outputWidth_ = (columns + PIXEL_PACKING - 1) / PIXEL_PACKING;

    switch (dataType_) {
        case param_type::WGT_FLOAT:
            break;
        default:
            isQuantized_ = true;
            break;
    }
#if defined(HIGH_PRECISION) || defined(__APPLE__)
    weightLanes_ = (opengl::GLInfo::getMaxVaryingVectors() >= 16) ? 2 : 1;
#else
    weightLanes_ = (opengl::GLInfo::getMaxVaryingVectors() >= 16) ? 4 : 2;
#endif
}

/**
 * @brief Destructor
 */
MatMulConst::~MatMulConst() {
    GLuint tex[4] = {weightData_, scaleData_, zeroData_, biasData_};
    glDeleteTextures(4, tex);
}

/**
 * @brief Generate proxy geometry and setup shaders
 */
void MatMulConst::setup() {
    CLEAR_GFXERR_DEBUG
    proxyGeometry();
    compileShaders();
}


/**
 * @brief Perform matrix multiplication
 *
 * @param dataRows Number of rows in the input sequence to multiply with (left matrix height)
 * @param outputRowOffset Starting row in the output sequence to write to
 * @param targetFBO Pointer to target FBO to write to
 *
 * @pre Source texture is bound to the appropriate unit (0), residual texture (if any) is
 *      bound to the appropriate unit (1), \c GL_SCISSOR_TEST is enabled.
 *
 * This function executes the matrix multiplication and writes the result to the target FBO.
 * Depending on the number of data rows, it selects between two different approaches to perform
 * the multiplication. For small numbers of data rows, it uses a short shader that performs
 * constant data-fetching (and dequantization) in the fragment shader. For larger number of rows,
 * the constant data-fetching happens in a vertex shader. See the detailed class documentation
 * for an explanation of these shaders.
 *
 * @warning This code currently only handles 4-bit quantized weight matrices
 *
 * @see weightMatMulShort4Bit, weightMatMulLong4Bit, MATMUL_LONG_THRESHOLD
 */
void MatMulConst::forward(int dataRows, int outputRowOffset, opengl::FBO *targetFBO) {
    CLEAR_GFXERR_DEBUG
    glLineWidth(1.0f);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    array_->bind();
    if (isQuantized_) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, weightData_);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, scaleData_);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, zeroData_);
    }
    if (hasBias_) {
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, biasData_);
    }
    glViewport(0, outputRowOffset, outputWidth_, dataRows);
    glScissor(0, outputRowOffset, outputWidth_, dataRows);
    if (dataRows >= MATMUL_LONG_THRESHOLD) {
        weightMatMulLong4Bit(targetFBO, dataRows);
    } else {
        weightMatMulShort4Bit(targetFBO, dataRows);
    }
    array_->unbind();

}


/**
 * @brief Place a custom shader into the pipeline
 *
 * @param shaderType Type of shader to be replaced (vertex/fragment long or short)
 * @param resource Resource name of the shader that is to replace the default shader
 *
 * This replaces the default shader identified by the \p shaderType with the shader that is
 * identified by the supplied \p resource name. Use this for (internal) customization purposes.
 *
 * @warning Shader customization/overrides use internal knowledge about how this operation works
 *          and may break at any time. Use at your own risk.
 */
void MatMulConst::customShader(shtype shaderType, const char *resource) {
    customShaders_[(int)shaderType] = resource;
}


/**
 * @brief Place a custom shader pre-processing function into the pipeline
 *
 * @param preprocFunc Function that performs the pre-processing
 *
 * Register a function that is called when shader preprocessor definitions are created. The
 * supplied function receives a pointer to the preprocessor buffer, its size and the associated
 * shader type and is to fill the preprocessing buffer appropriately.
 */
void MatMulConst::customShaderPreproc(const std::function<void(char *, size_t, shtype)> & preprocFunc) {
    customShaderPreproc_ = preprocFunc;
}

/**
 * @brief Place a custom shader post-processing function into the pipeline
 *
 * @param postFunc Function that performs post-linkage processing of the shader
 *
 * Register a function that is called once the shader(s) have been linked. The supplied function
 * receives a (raw) pointer to the linked shader as well as the type of shader to be post-processed.
 * The shader program will be bound already.
 */
void MatMulConst::customShaderPostproc(const std::function<void(opengl::ShaderProgram *, shtype)>& postFunc) {
    customShaderPost_ = postFunc;
}



/**
 * @brief Load matrix bias for this layer
 *
 * @param data Blob that contains the bias data
 *
 * This function parses the bias values stored in the \p data blob parameter for usage with the GPU.
*  It is presumed that this layer type performs an affine transformation of the input data by
 * using a \e left-multiplication of the form:
 *
 * \f[ \mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}\f]
 *
 * where \f$ x \in \mathbb{R}^{1 \times m} \f$, \f$ W \in \mathbb{R}^{m \times n} \f$ and
 * \f$ \mathbf{y}, \mathbf{b} \in \mathbb{R}^{1 \times n} \f$.
 *
 * It is assumed that the bias data is supplied as 32-bit floating-point data.
 *
 * @note It is safe to call this function from a context that is shared with the initial GL
 *       context that was used to create the layer.
 *
 *  @see loadWeights()
 */
void MatMulConst::loadBiases(const DataBlob & data) {
    // TODO (mw) check data format for float16 / float32
    if (hasBias_) {
        if (data.empty()) THROW_EXCEPTION_ARGS(FynException, "Bias data is empty for matrix multiplication");
        auto * ptr = std::any_cast<const float *>(data.get());
        glGenTextures(1, &biasData_);
        assert(biasData_ != 0);
        glBindTexture(GL_TEXTURE_2D, biasData_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        // FIXME (mw) handle round-off of channels here
        glTexImage2D(GL_TEXTURE_2D, 0, (GLint)GPULayerBase::TEXTURE_IFORMAT_4, outputWidth_, 1, 0,
                     (GLenum)GPULayerBase::TEXTURE_FORMAT_4, GL_FLOAT, ptr);
    }
}



/**
 * @brief Load matrix weights for this layer
 *
 * @param data Blob that contains the weight data
 *
 * This function parses the weights stored in the \p data blob  parameter for usage with the GPU.
*  It is presumed that this layer type performs an affine transformation of the input data by
 * using a \e left-multiplication of the form:
 *
 * \f[ \mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}\f]
 *
 * where \f$ x \in \mathbb{R}^{1 \times m} \f$, \f$ W \in \mathbb{R}^{m \times n} \f$ and
 * \f$ \mathbf{y}, \mathbf{b} \in \mathbb{R}^{1 \times n} \f$.
 *
 * Most importantly, the storage order of the supplied weights is supposed to be \b row-major
 * (i.e. the first \f$ m \f$ elements are the first row of \f$ \mathbf{W} \f$ and so forth).
 *
 * On \e quantized weights, we assume that quantization is done by packing data into 32-bit
 * words in an LSB-first fashion. To provide an example, when using 8-bit quantization we may
 * consider the 32-bit word as an array of 4 values. The byte that translates to array index
 * 0 would then be the \e lowest byte (also known as little-endian order), i.e. assuming an
 * array of the form:
 * @code
 * uint8_t array[4] = {0, 1, 2, 3}
 * @endcode
 *
 * This would be stored as \c 0x03020100 inside a single 32-bit word. For unknown reasons,
 * it is common practice for this quantized type of storage to have each 32-bit word represent
 * a <i>partial column</i>, i.e. the first \e n rows of a column and not the first \e n elements
 * of a row.
 *
 * @note It is safe to call this function from a context that is shared with the initial GL
 *       context that was used to create the layer.
 *
 * @warning See storage order assumption in the long description
 *
 *  @see loadBiases()
 */
void MatMulConst::loadWeights(const DataBlob & data) {
    if (data.empty()) THROW_EXCEPTION_ARGS(FynException, "Weight data is empty for matrix multiplication");
    glGenTextures(1, &weightData_);
    assert(weightData_ > 0);
    auto * ptr = std::any_cast<const uint8_t *>(data.get());
    gpu::rudiments::LinearTextureLoader::loadRM4BitQuantizedWeights(reinterpret_cast<const uint32_t *>(ptr), rows_, columns_, weightData_);
}


/**
 * @brief Load quantization tables
 *
 * @param scales DataBlob instance that holds the scale factors for the weight matrix, see long
 *               description for details
 *
 * @param zeros DataBlob instance that holds the zero points for the weight matrix, in \e quantized
 *              form using the same type of quantization as the weight matrix, see long description
 *              for details
 *
 * This function loads the quantization tables for this operation. The quantization tables are stored
 * in a per column fashion in relation to the weight matrix while assuming that the weight matrix
 * will be multiplied to the input from the \e right, i.e.:
 *
 * \f[ \mathbf{y} = \mathbf{x} \mathbf{W} \f]
 *
 * where \f$ \mathbf{x} \in \mathbb{R}^{1 \times m} \f$ , \f$ W \in \mathbb{R}^{m \times n} \f$ and
 * \f$ \mathbf{y} \in \mathbb{R}^{1 \times n} \f$ . Each column of the weight matrix will have
 * (at least) one scale factor and one zero point associated with it. In addition, we support
 * grouping of scales and zero points per column, effectively partitioning each column into a set
 * of row-segments with different scales and zero points.
 *
 * On an elementary level, the computation that is performed with quantized data is as follows:
 *
 * \f[ y_i = \sum_j x_j \cdot s \cdot \left( W_{ij} - (z+1) \right) \f]
 *
 * where we left out the grouping of \f$ s \f$ and \f$ z \f$ for simplicity.
 */
void MatMulConst::loadQuantizationTables(const DataBlob& scales, const DataBlob& zeros) {
    using namespace gpu::rudiments;
    assert(dataType_ != param_type::WGT_FLOAT);
    assert((rows_ % quantGroupSize_) == 0);
    GLuint tex[2];
    glGenTextures(2, tex);
    assert(tex[0] > 0 && tex[1] > 0);
    if (scales.get().type() == typeid(const float *)) {
        LinearTextureLoader::load4BitQuantizationTables<float, GL_R32F, GL_RED, GL_FLOAT>
                (std::any_cast<const float *>(scales.get()),
                 reinterpret_cast<const uint32_t *>(std::any_cast<const uint8_t *>(zeros.get())),
                 rows_, columns_, quantGroupSize_, tex[0], tex[1]);
    } else {
        LinearTextureLoader::load4BitQuantizationTables<uint16_t, GL_R16F, GL_RED, GL_HALF_FLOAT>
                (std::any_cast<const uint16_t *>(scales.get()),
                reinterpret_cast<const uint32_t *>(std::any_cast<const uint8_t *>(zeros.get())),
                rows_, columns_, quantGroupSize_, tex[0], tex[1]);
    }
    scaleData_ = tex[0];
    zeroData_ = tex[1];
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Perform matrix multiplication for long sequences (large(r) left matrix heights)
 *
 * @param target Pointer to FBO to render the data into
 * @param dataRows Number of rows in the left matrix to process
 *
 * Compute matrix/matrix multiplication for longer sequences using multiple passes and instanced
 * rendering on 4-bit quantized weight-matrices. Results are accumulated in the target FBO using
 * the ROPs blend functionality in a column-by-column order.
 */
void MatMulConst::weightMatMulLong4Bit(opengl::FBO *target, int dataRows) {
    if (rows_ % MMUL_WEIGHTS_PER_PASS) THROW_EXCEPTION_ARGS(FynException, "Number of rows (%d) must be a multiple of %d", rows_, MMUL_WEIGHTS_PER_PASS);
    int instances = ((PIXEL_PACKING / weightLanes_) * rows_) / MMUL_WEIGHTS_PER_PASS;
    target->bind();
    if (!outResidual_) glClear(GL_COLOR_BUFFER_BIT);
    if ((hasBias_) || (inResidual_)) {
        shaderLongPrime_->bind();
        shaderLongPrime_->setUniformVec2("viewport", outputWidth_, dataRows);
        shaderLongPrime_->setUniformValue("quantGroupSize", quantGroupSize_);
        glDrawArrays(GL_LINES, 0, outputWidth_ * 2);
        shaderLongPrime_->unbind(true);
        instances -= 1;
        glActiveTexture(GL_TEXTURE0 + BIAS_UNIT);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0 + RESIDUAL_UNIT);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    shaderLong_->bind();
    shaderLong_->setUniformVec2("viewport", outputWidth_, dataRows);
    shaderLong_->setUniformValue("quantGroupSize", quantGroupSize_);
    glDrawArraysInstanced(GL_LINES, 0, outputWidth_ * 2, instances);
    target->unbind();
    shaderLong_->unbind();
}


/**
 * @brief Perform matrix multiplication for short sequences (small(er) left matrix heights)
 *
 * @param target Pointer to FBO to render the data into
 * @param dataRows Number of rows in the left matrix to process
 *
 * Compute matrix/matrix multiplication for short sequences using multiple passes and instanced
 * rendering on 4-bit quantized weight-matrices. Results are accumulated in the target FBO using
 * the ROPs blend functionality.
 */
void MatMulConst::weightMatMulShort4Bit(opengl::FBO *target, int dataRows) {
    int div = MMUL_WEIGHTS_PER_PASS * smallMWPacks_;
    if (rows_ % div) THROW_EXCEPTION_ARGS(FynException, "Number of rows (%d) must be a multiple of %d", rows_, div);
    int instances = rows_  / div;
    target->bind();
    if (!outResidual_) glClear(GL_COLOR_BUFFER_BIT);
    if ((hasBias_) || (inResidual_)) {
        shaderShortPrime_->bind();
        shaderShortPrime_->setUniformVec2("viewport", outputWidth_, dataRows);
        shaderShortPrime_->setUniformValue("quantGroupSize", quantGroupSize_);
        glDrawArrays(GL_LINES, 0, dataRows * 2);
        shaderShortPrime_->unbind(true);
        instances -= 1;
        glActiveTexture(GL_TEXTURE0 + BIAS_UNIT);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0 + RESIDUAL_UNIT);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    shaderShort_->bind();
    shaderShort_->setUniformVec2("viewport", outputWidth_, dataRows);
    shaderShort_->setUniformValue("quantGroupSize", quantGroupSize_);
    glDrawArraysInstanced(GL_LINES, 0, dataRows * 2, instances);
    target->unbind();
    shaderShort_->unbind();
}


/**
 * @brief Compile vertex- and fragment shaders that carry out the computation
 */
void MatMulConst::compileShaders() {
    // TODO (mw) support for other quantizations and no quantizations
    using namespace opengl;
    char preproc[1024] = {0};
    bool isprimed = (hasBias_ || inResidual_);
    const char * shortvert = (customShaders_[VERT_SHORT]) ? customShaders_[VERT_SHORT] : "shaders/sequence/seq_matmul_4bit_short.vert";
    const char * shortfrag = (customShaders_[FRAG_SHORT]) ? customShaders_[FRAG_SHORT] : "shaders/sequence/seq_matmul_4bit_short.frag";
    snprintf(preproc, sizeof(preproc) - 1,"#define MATRIX_WEIGHTS %d\n#define MATRIX_PACKS %d\n#define INSTANCE_OFFSET %d\n", MMUL_WEIGHTS_PER_PASS / PIXEL_PACKING, smallMWPacks_, (isprimed) ? 1 : 0);
    if (customShaderPreproc_) customShaderPreproc_(preproc, sizeof(preproc) - strlen(preproc) - 1, shtype::ANY_SHORT);
    preamble_.generatePreprocessorPreamble(preproc, sizeof(preproc) - strlen(preproc)-1, LayerFlags::RESIDUAL_INPUT);
    shaderShort_ = ShaderRepository::compileShaderPair(shortvert, shortfrag,preproc, typeid(this), context());
    shaderShort_->bindAttributeLocation("attributes0", 0);
    shaderShort_->link();
    postProcessShader(shaderShort_.get(), shtype::ANY_SHORT);
    if ((hasBias_) || (inResidual_)) {
        snprintf(preproc, sizeof(preproc) - 1, "#define MATRIX_WEIGHTS %d\n#define MATRIX_PACKS %d\n", MMUL_WEIGHTS_PER_PASS / PIXEL_PACKING, smallMWPacks_);
        if (hasBias_) strncat(preproc, "#define USE_BIAS\n", sizeof(preproc)-1);
        if (inResidual_) strncat(preproc, "#define USE_RESIDUAL\n", sizeof(preproc)-1);
        preamble_.generatePreprocessorPreamble(preproc, sizeof(preproc) - strlen(preproc)-1, LayerFlags::RESIDUAL_INPUT);
        shaderShortPrime_ = ShaderRepository::compileShaderPair(shortvert, shortfrag, preproc, typeid(this), context());
        shaderShortPrime_->bindAttributeLocation("attributes0", 0);
        shaderShortPrime_->link();
        assert(glGetError() == GL_NO_ERROR);
        postProcessShader(shaderShortPrime_.get(), shtype::ANY_SHORT);
        assert(glGetError() == GL_NO_ERROR);
    }
    // FIXME (mw) we should do a runtime-check instead to see if the GLSL version fits
#if defined(HIGH_PRECISION) || defined(__APPLE__)
    const char * longvert = (customShaders_[VERT_LONG]) ? customShaders_[VERT_LONG] : "shaders/sequence/seq_matmul_4bit_long.vert";
    const char * longfrag = (customShaders_[FRAG_LONG]) ? customShaders_[FRAG_LONG] : "shaders/sequence/seq_matmul_4bit_long.frag";
#else
    const char * longvert = (customShaders_[VERT_LONG]) ? customShaders_[VERT_LONG] : "shaders/sequence/seq_matmul_4bit_long_half.vert";
    const char * longfrag = (customShaders_[FRAG_LONG]) ? customShaders_[FRAG_LONG] : "shaders/sequence/seq_matmul_4bit_long_half.frag";
#endif
    snprintf(preproc, sizeof(preproc) - 1, "#define MATRIX_WEIGHTS %d\n#define NUM_LANES %d\n#define INSTANCE_OFFSET %d\n", MMUL_WEIGHTS_PER_PASS / PIXEL_PACKING, weightLanes_, (isprimed) ? 1: 0);
    if (customShaderPreproc_) customShaderPreproc_(preproc, sizeof(preproc) - strlen(preproc) - 1, ANY_LONG);
    preamble_.generatePreprocessorPreamble(preproc, sizeof(preproc) - strlen(preproc)-1, LayerFlags::RESIDUAL_INPUT);
    shaderLong_ = ShaderRepository::compileShaderPair(longvert, longfrag,preproc, typeid(this), context());
    shaderLong_->bindAttributeLocation("attributes0", 0);
    shaderLong_->link();
    if ((hasBias_) || (inResidual_)) {
        snprintf(preproc, sizeof(preproc) - 1, "#define MATRIX_WEIGHTS %d\n#define NUM_LANES %d\n", MMUL_WEIGHTS_PER_PASS / PIXEL_PACKING, weightLanes_);
        if (hasBias_) strncat(preproc, "#define USE_BIAS\n", sizeof(preproc)-1);
        if (inResidual_) strncat(preproc, "#define USE_RESIDUAL\n", sizeof(preproc)-1);
        preamble_.generatePreprocessorPreamble(preproc, sizeof(preproc) - strlen(preproc)-1, LayerFlags::RESIDUAL_INPUT);
        shaderLongPrime_ = ShaderRepository::compileShaderPair(longvert, longfrag, preproc, typeid(this), context());
        shaderLongPrime_->bindAttributeLocation("attributes0", 0);
        shaderLongPrime_->link();
        postProcessShader(shaderLongPrime_.get(), shtype::ANY_LONG);
        assert(glGetError() == GL_NO_ERROR);
    }
    postProcessShader(shaderLong_.get(), shtype::ANY_LONG);
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @brief Run post-processing on the supplied shader
 *
 * @param shader Pointer to (linked) shader program
 * @param type Shader type
 *
 * Runs some post-processing on the shader, mainly binding related for older GL versions. Will
 * call the #customShaderPost_ function if set.
 *
 * @see shtype, setCustomShaderPost
 */
void MatMulConst::postProcessShader(ShaderProgram * shader, shtype type) {
    assert(shader->isLinked());
    shader->bind();
    if (customShaderPost_) customShaderPost_(shader, type);
    else if (!opengl::GLInfo::hasBinding()) {
        shader->setUniformValue("inputLayer0", 0);
        shader->setUniformValue("inputLayer1", 1, true);
        shader->setUniformValue("matrix", 2);
        shader->setUniformValue("scaleData", 3);
        shader->setUniformValue("zeroData", 4);
        shader->setUniformValue("biasData", 5, true);
        shader->setUniformValue("residual", 6, true);
    }
    shader->unbind();
}


 /**
  * @brief Generate proxy geometry for the operation
  */
void MatMulConst::proxyGeometry() {
    using namespace opengl;
     array_ = std::make_unique<VAO>(context());
     array_->bind();
     int numlines = std::max(outputWidth_, outputHeight_);
     auto * attrs0 = new uint32_t[numlines * 2];
     for  (int line=0, offset=0; line < numlines; line++) {
         attrs0[offset++] = (line << 16);
         attrs0[offset++] = (line << 16) | 1;
     }
     vertices_ = std::make_unique<VBO>(context());
     array_->enableArray(0);
     vertices_->setBufferData(attrs0, (int)(numlines * 2 * sizeof(uint32_t)), GL_STATIC_DRAW);
     vertices_->bind();
     array_->setVertexAttributeBuffer(0, 1, GL_UNSIGNED_INT, 0, 0);
     delete [] attrs0;
     array_->unbind();
     vertices_->unbind();
}


} // rudiments namespace

// vim: set expandtab ts=4 sw=4:
