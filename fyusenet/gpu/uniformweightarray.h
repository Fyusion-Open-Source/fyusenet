//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform (GL) Weight Array (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/shaderprogram.h"
#include "../base/layerflags.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Class to encapsulate a pair of network weights and bias/bn values for a convolutional network layer
 *
 * This class serves as a base class to store weight/bias (and batchnorm) data to be used for
 * convolutional layers.
 *
 * As FyuseNet performs computation using fragment shaders and polygons, we either have to use
 * textures or uniform values as data sources. Benchmarking on various (mobile) GPUs resulted in
 * favour of using uniform variables inside the fragment shaders. Most likely because these
 * are stored in constant memory of the shader units which have similar/identical speed as
 * a register access.
 *
 * In any case, this class stores the convolution weights separate from the bias and batchnorm
 * values and arranges them in \e packages. A package is one block of weight/bias/bn values
 * that is required for one or multiple shader passes. The number of shader passes required to
 * perform a convolution depends on:
 *   - kernel size
 *   - number of input channels
 *   - number of output channels
 *   - individual implementation of the convolution shader
 *
 * The definition of a package is dependent on the type of convolution layer and varies
 * in the implementations. See ConvWeightArrayKxKxNxM and DepthwiseConvWeightArrayKxKxNxM
 * for specifics.
 *
 * @see ConvWeightArrayKxKxNxM, DepthwiseConvWeightArrayKxKxNxM
 */
class UniformWeightArray {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    UniformWeightArray();
    virtual ~UniformWeightArray();

    // ------------------------------------------------------------------------
    // Public methods (interface)
    // ------------------------------------------------------------------------

    /**
     * @brief Retrieve pointer to weight data usable for uploading them into a uniform array
     *
     * @param inputPass The input batch (offset to the input-shader-pass batch to process)
     * @param outputPass The output batch (offset to the output-shader-pass batch to process)
     * @param xIndex The KxK convolution kernel filter spatial x-index (in 0..K-1)
     * @param yIndex The KxK convolution kernel filter spatial y-index (in 0..K-1)
     *
     * @return Pointer to contiguous float data which can be loaded as uniform float data to a
     *         OpenGL shader
     *
     * This function returns a pointer to the selected weight package. See the class description
     * for more information on packages and check the derived classes for the specifics.
     *
     * @note It should be noted that this function returns the pointer for the first render target
     *       of a render pass. If individual render targets for a single pass are to be addressed,
     *       it is up to the caller to sort out the required pointer arithmetic.
     */
    virtual const float * getPackageWeights(int inputPass, int outputPass, int xIndex, int yIndex) const = 0;

    /**
     * @brief Retrieve pointer to bias usable for uploading them into a uniform array
     *
     * @param outputPass The output batch (offset to the output-shader-pass batch to process)
     *
     * @return Pointer to bias data (sized by the number of render targets for this specific
     *         output pass)
     */
    virtual const float * getPackageBias(int outputPass) const = 0;

    /**
     * @brief Retrieve pointer to batchnorm scale values usable for uploading them into a uniform array
     *
     * @param outputPass The output batch (offset to the output-shader-pass batch to process)
     *
     * @return Pointer to batchnorm (scale only) data, suitably sized for the number of render
     *         targets in the specified \p outputPass
     *
     * This function is used when using a post-batchnorm approach to apply the batchnorm. In that
     * case, the shader applies the batchnorm after the convolution and before the bias (which has
     * been pre-scaled already by a derived class of this class).
     */
    virtual const float * getPackageBNScale(int outputPass) const = 0;

    /**
     * @brief Extract bias data from raw input data
     *
     * @param input Pointer to raw input data
     * @param offset Offset (in 32-bit floats) to \p input where to start reading the bias data
     *
     * @todo Support 16-bit FP and perhaps 8/16 bit integer raw data in the future
     *
     * This copies the bias data, which is presumed to be a simple contiguous array of 32-bit
     * floating-point values, from the raw data \p input to internal memory.
     *
     * @note Please note the input format description in the class documentation
     */
    virtual void extractBiasData(const float *input, size_t offset)=0;

    /**
     * @brief Extract weight data from raw input data
     *
     * @param input Pointer to raw input data
     * @param offset Offset (in 32-bit floats) to \p input where to start reading the weights,
     *               <i>pointing to the start of the weight data</i>
     *
     * This function extracts the convolution weights from the supplied raw \p input pointer,
     * assuming that it points to the start of the actual weight data.
     * We assume that the weight data is laid out in a multidimensional array with data arranged
     * like (inner to outer):
     *  1. input-channel
     *  2. kernel-x
     *  3. kernel-y
     *  4. output-channel
     *
     *  Or in C array notation:
     *  @code
     *  [outchannel][fy][fx][inchannel]
     *  @endcode
     */
    virtual void extractWeightData(const float *input, size_t offset)=0;

    /**
     * @brief Extract batchnorm data from raw input data
     *
     * @param input Pointer to raw input data
     * @param offset Offset (in 32-bit floats) to \p input where to start reading the batchnorm
     *               data
     *
     * This function assumes that the batchnorm data in the \p input pointer is arranged in the
     * following way:
     *   1. scales (one 32-bit FP per output channel)
     *   2. offsets (one 32-bit FP per output channel)
     */
    virtual void extractBatchnormData(const float *input, size_t offset)=0;

    /**
     * @brief Retrieve the number of required input render passes for one input batch
     *
     * @return Number of input render passes
     *
     * The number of input render passes is usually the ceiling of the number of channels
     * divided by 4.
     */
    virtual int numInputRenderPasses() const=0;

    /**
     * @brief Retrieve the number of required output render passes for one input batch
     *
     * @return Number of output render passes (including compensation for multiple render targets)
     *
     * This returns the number of output render passes required. Due to the potential use of
     * multiple render targets, this may not be simply calculated the same way as the number
     * of input render passes. Instead this takes into account the number of render targets
     * for each render pass.
     *
     * @see numRenderTargets()
     */
    virtual int numOutputRenderPasses() const=0;

    /**
     * @brief Retrieve number of render targets for
     *
     * @param outputPass The output batch (offset to the output-shader-pass batch to process)
     *
     * @return Number of render targets for the specified \p outputPass
     */
    virtual int numRenderTargets(int outputPass) const=0;

    /**
     * @brief Retrieve offset for output textures based on rendering pass
     *
     * @param outputPass The output batch (offset to the output-shader-pass batch to process)
     *
     * @return Texture index (starting at 0) which is used as the first texture in the suplied
     *         \p outputPass
     *
     * This function is useful in conjunction with residual channels, as there are as many
     * residual channels as there are output channels and the multiple render targets
     * necessitate to perform a lookup which output pass starts at what texture.
     */
    virtual int outputTextureOffset(int outputPass) const=0;

    /**
     * @brief Retrieve size (in number of 32-bit FP elements) of convolution weight package
     *
     * @param inputPass
     * @param outputPass The output batch (offset to the output-shader-pass batch to process)
     * @param xIndex The KxK convolution kernel filter spatial x-index (in 0..K-1)
     * @param yIndex The KxK convolution kernel filter spatial y-index (in 0..K-1)
     *
     * @return Number of 32-bit FP elements inside the convolution weight package for the supplied
     *         parameters (excluding size of bias/BN)
     */
    virtual int getPackageSize(int inputPass, int outputPass, int xIndex, int yIndex) const=0;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Set user-specific data to be stored with the weight array
     *
     * @param userData Pointer to user-specific data
     *
     * Weight arrays allow for storing user-specific data, for example to uniquely identify
     * an array or link to other data structures. User-specified data is untyped and the
     * weight array instance does not take ownership.
     */
    void setUserData(void *userData) {
      userData_ = userData;
    }

    /**
     * @brief Retrieve pointer to user-specific data stored in the weight array
     *
     * @return Pointer to user-specific data, may be \c nullptr
     */
    void * getUserData() const {
      return userData_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------

    /**
     * @copydoc gpu::PIXEL_PACKING
     */
    static constexpr int PIXEL_PACKING = gpu::PIXEL_PACKING;

    void *userData_ = nullptr;                  //!< Pointer to optional (untyped) user data
    float *weightData_ = nullptr;               //!< Pointer to convolution weight data (32-bit FP)
    float *biasData_ = nullptr;                 //!< Pointer to bias data (32-bit FP)
    float *bnBias_ = nullptr;                   //!< Pointer to batchnorm offset/bias (32-bit FP)
    float *bnScale_ = nullptr;                  //!< Pointer to batchnorm scales (32-bit FP)
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
