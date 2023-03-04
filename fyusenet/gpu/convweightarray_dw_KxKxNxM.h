//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform Array for Depthwise Convolutional Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "uniformweightarray.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Weight array for depthwise KxK convolution using 4-elements per pixel
 *
 * This class implements the UniformWeightArray interface for "depthwise" convolutions under
 * varying kernel sizes and input/output channel configurations.
 *
 * An object of this class stores the convolution weights separate from the bias and batchnorm
 * values and arranges them in \e packages. A package is one block of weight/bias/bn values
 * that is required for a single shader pass. The number of shader passes required to perform
 * a convolution depends on:
 *   - kernel size
 *   - number of input channels
 *   - number of output channels
 *
 * As a general rule, packages can be viewed as simple multidimensional arrays, with 4 weights
 * for 4 consecutive input and output channels as the basic element (also see the PIXEL_PACKING
 * value). The weights are stored contiguously as nested array in the following order (from inner
 * to outer):
 *   1. x-axis of the kernel
 *   2. y-axis of the kernel
 *   3. render pass size (see below)
 *   4. passes
 *
 * Or in nested arrays notation:
 * @code
 * [pass][psize][ky][kx]
 * @endcode
 *
 * The current code assumes that there are always as many input channels as there are output
 * channels and therefore there is no distinction between input and output pass.
 *
 * @todo Code is currently quite constrained in terms of usable channel multipliers (group size),
 *       improve on that in future versions.
 */
class DepthwiseConvWeightArrayKxKxNxM : public UniformWeightArray {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DepthwiseConvWeightArrayKxKxNxM(int kernel, int inputChannels, int channelMultiplier, int maxRenderTargets, int maxTextures);
    virtual ~DepthwiseConvWeightArrayKxKxNxM();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual const float * getPackageWeights(int inputPass, int outputPass, int xIndex, int yIndex) const override;
    virtual const float * getPackageBias(int outputPass) const override;
    virtual const float * getPackageBNScale(int outputPass) const override;
    virtual void extractBiasData(const float *input, size_t offset) override;
    virtual void extractWeightData(const float *input, size_t offset) override;
    virtual void extractBatchnormData(const float *input, size_t offset) override;
    virtual int numInputRenderPasses() const override;
    virtual int numOutputRenderPasses() const override;
    virtual int numRenderTargets(int outputPass) const override;
    virtual int outputTextureOffset(int outputPass) const override;
    virtual int getPackageSize(int inputPass, int outputPass, int xIndex, int yIndex) const override;
 private:

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int kernel_ = 0;                                       //!< Spatial kernel size (isotropic)
    int channelMultiplier_ = 1;
    int maxRenderTargets_ = 0;                             //!< Maximum number of render targets for a single pass
    int maxTextures_ = 0;                                  //!< Maximum number of input textures (currently unused)
    int inputChannels_ = 0;                                //!< Number of total input channels
    int outputChannels_ = 0;                               //!< Number of total output channels (after applying the channel multiplier)
    int paddedInputChannels_ = 0;                          //!< Number of input channels padded to next multiple of #PIXEL_PACKING
    int paddedOutputChannels_ = 0;                         //!< Number of output channels padded to next multiple of #PIXEL_PACKING
    int outputRenderPasses_ = 0;                           //!< Total number of render passes required to cover all output layers
    std::vector<int> MRT_;                                 //!< Stores the number of render-targets per output pass
    std::vector<int> MRTOffsets_;                          //!< Stores the output channel offsets
    std::vector<int> MRTChannels_;                         //!< Channel multiplier index (untested)
    unsigned int *packOffsets_ = nullptr;                  //!< Offsets for packages in #weightData_ block
    unsigned int *packSizes_ = nullptr;                    //!< Package size (in elemnets) for each package in the #weightData_ block
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
