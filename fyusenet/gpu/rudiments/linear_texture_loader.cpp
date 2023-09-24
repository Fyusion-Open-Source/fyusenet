//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GL Texture Loader for Linear Data                                           (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/glinfo.h"
#include "linear_texture_loader.h"
#include "../../common/miscdefs.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::rudiments {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Load attention matrix weights for this layer
 *
 * @param weights Pointer to array with weight values (see long description)
 *
 * @param rows Rows of the weight matrix to load
 *
 * @param columns Columns of the weight matrix to load
 *
 * @param wgtTex OpenGL texture handle to place the weights in
 *
 * This function parses the weights stored in the \p weights parameter for usage with the GPU.
 * It is presumed that layers using the texture perform linear transformations of the input data
 * by using a \e left-multiplication of the form:
 *
 * \f[ \mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}\f]
 *
 * where \f$ x \in \mathcal{R}^{1 \times m} \f$, \f$ W \in \mathcal{R}^{m \times n} \f$ and
 * \f$ \mathbf{y}, \mathbf{b} \in \mathcal{R}^{1 \times n} \f$.
 *
 * Most importantly, the storage order of the supplied weights is supposed to be \b row-major
 * (i.e. the first \f$ m \f$ elements are the first row of \f$ \mathbf{W} \f$ and so forth).
 *
 * On \e quantized weights, we assume that an <i>affine quantization mapping</i> is used and
 * quantization is done by packing data into 32-bit words in an LSB-first fashion. To provide an
 * example, when using 8-bit quantization we may consider the 32-bit word as an array of 4 values.
 * The byte that translates to array index 0 would then be the \e lowest byte (also known as
 * little-endian order), i.e. assuming an array of the form:
 * @code
 * uint8_t array[4] = {0, 1, 2, 3}
 * @endcode
 *
 * This would be stored as \c 0x03020100 inside a single 32-bit word. For unknown reasons
 * it is common practice for this quantized type of storage to have each 32-bit word represent
 * a <i>partial column</i>, i.e. the first \e n rows of a column and not the first \e n elements
 * of a row.
 *
 * @note It is safe to call this function from a context that is shared with the initial GL
 *       context that was used to create the layer. Also note that the data is converted into
 *       column-major order during loader (requiring a transform buffer dynamically allocated
 *       and deallocated in this function)
 *
 * @warning See input storage order assumption in the long description
 */
void LinearTextureLoader::loadRM4BitQuantizedWeights(const uint32_t * weights, int rows, int columns, GLuint wgtTex) {
    assert(wgtTex > 0);
#ifdef DEBUG
    glGetError();
#endif
    // -----------------------------------------------------------
    // Unpack the weight data into the format that we use in the
    // (GL) shaders...
    // -----------------------------------------------------------
    auto * tmp = new uint32_t[(rows * columns) / 8 + 1];
    int rows4 = (rows + 7) / 8;
    for (int col=0,tgtoffs=0; col < columns; col++) {
        for (int row=0, ridx=0; row < rows4; row++, ridx += columns) {
            tmp[tgtoffs++] = weights[ridx + col];
        }
    }
#ifdef GL_R32UI
    const GLint wgtiformat = GL_RGBA32UI;
    const GLenum wgtformat = GL_RGBA_INTEGER;
#else
    const GLint wgtiformat = GL_RGBA32UI_EXT;
    const GLenum wgtformat = GL_RGBA_INTEGER_EXT;;
#endif
    bindTexture(wgtTex);
    // NOTE (mw) we store the matrix in column-major order, therefore transpose the texture
    glTexImage2D(GL_TEXTURE_2D, 0, wgtiformat, (rows4 + 3) / 4, columns, 0, wgtformat, GL_UNSIGNED_INT, tmp);
#ifdef DEBUG
    assert(glGetError() == GL_NO_ERROR);
#endif
    delete [] tmp;
}


/**
 * @brief Load quantization tables for this layer
 *
 * @param scales Pointer to quantization scale data for affine quantization mapping
 * @param qZeros Pointer to quantization zero-offsets for affine quantization mapping
 * @param rows Number of rows in the weight matrix
 * @param columns Number of columns in the weight matrix
 * @param quantGroupSize Quantization group size
 * @param scaleTex GL texture ID for the quantization scales
 * @param zeroTex GL texture ID for the zero-offsets
 *
 * @tparam T Data type of the quantization scales on the CPU
 * @tparam gpulayout GL internal texture format of the scales
 * @tparam cpulayout GL texture format of the scales on the CPU
 * @tparam cputype GL datatype of the scales on the CPU
 *
 * This function loads the quantization tables for the quantized linear mapping.
 */
template<typename T, GLint gpulayout, GLenum cpulayout, GLenum cputype>
void LinearTextureLoader::load4BitQuantizationTables(const T * scales, const uint32_t *qZeros,
                                                     int rows, int columns, int quantGroupSize,
                                                     GLuint scaleTex, GLuint zeroTex) {
    assert(scaleTex > 0);
    assert(zeroTex > 0);
    CLEAR_GFXERR_DEBUG
    // -----------------------------------------------------------
    // The scale data is stored on a "per column" basis, where the
    // quantization group size allows for more than one value per
    // column.
    // -----------------------------------------------------------
    bindTexture(scaleTex);
    if (columns > opengl::GLInfo::getMaximumTextureSize()) {
        THROW_EXCEPTION_ARGS(opengl::GLException, "Texture size %d exceeds maximum system texture size (%d)", columns, opengl::GLInfo::getMaximumTextureSize());
    }
    glTexImage2D(GL_TEXTURE_2D, 0, gpulayout, columns, (rows + quantGroupSize - 1) / quantGroupSize,
                 0, cpulayout, cputype, scales);
#ifdef DEBUG
    assert(glGetError() == GL_NO_ERROR);
#endif
    // -----------------------------------------------------------
    // and finally the zero point data (we leave it packed on the
    // GPU, though it might be a better idea to unpack it before).
    // It is supplied in CPU memory as 32-bit integers, with 8
    // (quantized) entries per 32-bit integer. Each entry belongs
    // to a quantization group and the data is stored in row-
    // major order where (as opposed to the weights) each entry
    // in the 8-tuple belongs to a different column.
    // -----------------------------------------------------------
    bindTexture(zeroTex);
#ifdef GL_R32UI
    const GLint ziformat = GL_R32UI;
    const GLenum zformat = GL_RED_INTEGER;
#else
    const GLint ziformat = GL_R32UI_EXT;
    const GLenum zformat = GL_RED_INTEGER_EXT;;
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, ziformat,
                 std::max(1, (columns+7)/8), (rows + (quantGroupSize - 1)) / quantGroupSize,
                 0, zformat, GL_UNSIGNED_INT, qZeros);
#ifdef DEBUG
    assert(glGetError() == GL_NO_ERROR);
#endif
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Bind texture for loading data into it (and parameterize it)
 *
 * @param texture GL texture ID to bind and parameterize
 */
void LinearTextureLoader::bindTexture(GLuint texture) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/*##################################################################################################
#                  E X P L I C I T   T E M P L A T E    I N S T A N T I A T I O N S                #
##################################################################################################*/

template
void LinearTextureLoader::load4BitQuantizationTables<float, GL_R32F, GL_RED, GL_FLOAT>(const float * scales, const uint32_t *qZeros,
                                                               int rows, int columns, int quantGroupSize,
                                                               GLuint scaleTex, GLuint zeroTex);

template
void LinearTextureLoader::load4BitQuantizationTables<uint16_t, GL_R16F, GL_RED, GL_HALF_FLOAT>(const uint16_t * scales, const uint32_t *qZeros,
                                                                           int rows, int columns, int quantGroupSize,
                                                                           GLuint scaleTex, GLuint zeroTex);

} // fyusion::fyusenet::gpu::rudiments namespace

// vim: set expandtab ts=4 sw=4:
