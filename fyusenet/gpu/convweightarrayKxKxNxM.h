//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform Array for Convolutional Layer KxKxNxM Weights Shader (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "uniformweightarray.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Weight array for KxK convolution using 4-elements per pixel
 *
 * This class implements the UniformWeightArray interface for "standard" convolutions under
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
 * As a general rule, packages can be viewed as simple multidimensional arrays, with 16 weights
 * for 4 consecutive input and output channels as the basic element (also see the PIXEL_PACKING
 * value). These 16 weights can be viewed as a 4x4 matrix which multiplies and adds 4 input channels
 * for 4 output channels. The weights are stored contiguously as nested array in the following order
 * (from inner to outer):
 *
 *   1. x-axis of the kernel
 *   2. render pass size (see below)
 *   3. y-axis of the kernel
 *   4. input passes
 *   5. output passes
 *
 * Or in nested arrays notation:
 * @code
 * [outpass][inchan][ky][psize][kx]
 * @endcode
 *
 * The number of render passes and the individual size are controlled by the constructor, via
 * the total number of output channels and the maximum number of render targets. In the most
 * simple setup with a single render target, the array order basically becomes equivalent to
 *
 * @code
 * [outchan][inchan][ky][kx]
 * @endcode
 *
 * But as we support multi-render targets during rendering, a set of more than 4 output channels can
 * be rendered into at the same time. The maximum number of render targets is determined by system
 * limits, as well as by the convolution layer code itself which bases the decision on kernel size
 * and system specifics.
 *
 * @see UniformWeightArray
 */
class ConvWeightArrayKxKxNxM : public UniformWeightArray {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ConvWeightArrayKxKxNxM(int kernel, int inputChannels, int outputChannels, int maxRenderTargets);
    virtual ~ConvWeightArrayKxKxNxM();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual const float * getPackageWeights(int inputPass,int outputPass,int xIndex,int yIndex) const override;
    virtual const float * getPackageBias(int outputPass) const override;
    virtual const float * getPackageBNScale(int outputPass) const override;
    virtual void extractBiasData(const float *input, size_t offset) override;
    virtual void extractWeightData(const float *input, size_t offset) override;
    virtual void extractBatchnormData(const float *input, size_t offset) override;
    virtual int numInputRenderPasses() const override;
    virtual int numOutputRenderPasses() const override;
    virtual int numRenderTargets(int outputPass) const override;
    virtual int outputTextureOffset(int outputPass) const override;
    virtual int getPackageSize(int inputPass, int outputPass, int xindex, int yindex) const override;
 private:

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int kernel_ = 0;                                    //!< Kernel size (we only support isotropic kernels)
    int groupSize_ = 1;                                 //!< Group size for grouped convolutions
    int maxRenderTargets_ = 0;                          //!< Maximum number of render targets for a single pass
    int paddedInputChannels_ = 0;                       //!< Number of input channels padded to a multiple of #PIXEL_PACKING
    int inputChannels_ = 0;                             //!< Number of total input channels
    int outputChannels_ = 0;                            //!< Number of total output channels
    int paddedOutputChannels_ = 0;                      //!< Number of output channels padded to next multiple of #PIXEL_PACKING
    int inputRenderPasses_ = 0;                         //!< Total number of render passes required to cover all input channels
    int outputRenderPasses_ = 0;                        //!< Total number of render passes required to cover all output channels
    std::vector<int> MRT_;                              //!< Stores the number of render-targets per output pass
    std::vector<int> MRTOffsets_;                       //!< Stores the output channel offsets
    unsigned int *packOffsets_ = nullptr;               //!< Offsets for packages in #weightData_ block
    unsigned int *packSizes_ = nullptr;                 //!< Package size (in elemnets) for each package in the #weightData_ block
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
