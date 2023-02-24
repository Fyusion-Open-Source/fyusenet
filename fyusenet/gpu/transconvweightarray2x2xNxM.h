//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform Array for Transpose-Convolutional Layer 2x2xNxM Weights Shader (Header)
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
 * @brief Weight array for transpose 2x2 convolution using 4-elements per pixel
 *
 * This class reformats the input weight/bias/bn data for use in transpose convolutions using a
 * 2x2 convolution kernel. In contrast to standard convolutional layers, the transposed convolution
 * is often used for upsampling purposes (sometimes called deconvolution) performs a "broadcasting"
 * operation on the input tensor, akin to a Kronecker product, by multiplying the kernel with each
 * element in the input tensor and adding it to the output tensor. When performing upsampling, the
 * upsampling stride determines the spacing between the multiplied kernel elements in the output tensor.
 *
 * An implementation of a transpose convolution in a fragment shader is a tiny bit tricky due to the
 * broadcasting nature of the operator. The implementations in FyuseNet use a stencil buffer for the
 * broadcasting operation. Currently the transpose convolution layers in FyuseNet only support
 * stride-2 transpose-convolutions, which performs a "convoluted upsampling" of the input tensor by
 * a factor of 2 along both spatial dimensions. The fixed 2-fold upsampling basically leads to 4
 * different configurations which are encoded in a stencil-buffer and 4 specialized shaders for
 * each of the configurations. These configurations are referred to as stratum / strata internally.
 *
 * @see TransConvLayer2x2
 */
// TODO (mw) consolidate with 3x3 transconv weightarray to a more general class
class TransConvWeightArray2x2xNxM : public UniformWeightArray {
 public:
    enum {
        MAX_OUTPUT_CHANNELS = 1024,
        STRATA = 4
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    TransConvWeightArray2x2xNxM(int stride, int inputChannels, int outputChannels, int maxRenderTargets);
    virtual ~TransConvWeightArray2x2xNxM();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual const float * getPackageWeights(int inputPass, int outputPass, int xIndex, int yIndex) const override;
    virtual const float * getPackageBias(int outputPass) const override;
    virtual const float * getPackageBNScale(int outputPass) const override;
    virtual void extractBiasData(const float *input, size_t offset) override;
    virtual void extractWeightData(const float *input, size_t offset) override;
    virtual void extractBatchnormData(const float *input, size_t offset) override;
    virtual int numOutputRenderPasses() const override;
    virtual int numInputRenderPasses() const override;
    virtual int numRenderTargets(int outputPass) const override;
    virtual int outputTextureOffset(int outputPass) const override;
    virtual int getPackageSize(int inputPass, int outputPass, int xindex, int yindex) const override;
 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    int extractStratum1(const float *input, size_t inputOffset, int dstOffset);
    int extractStratum2(const float *input, size_t inputOffset, int dstOffset);
    int extractStratum3(const float *input, size_t inputOffset, int dstOffset);
    int extractStratum4(const float *input, size_t inputOffset, int dstOffset);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int kernel_;                          //!< Kernel size (we currently only support 2x2 kernels)
    int stride_;                          //!< Convolution stride (upsampling factor, we currently only support a stride of 2)
    int maxRenderTargets_;                //!< Maximum number of render targets for a single pass
    int paddedInputChannels_;             //!< Number of input channels padded to a multiple of #PIXEL_PACKING
    int inputChannels_;                   //!< Number of total input channels
    int outputChannels_;                  //!< Number of total output channels
    int paddedOutputChannels_;            //!< Number of output channels padded to next multiple of #PIXEL_PACKING
    int inputRenderPasses_;               //!< Total number of render passes required to cover all input channels
    int outputRenderPasses_;              //!< Total number of render passes required to cover all output channels
    std::vector<int> MRT_;                //!< Stores the number of render-targets per output pass
    std::vector<int> MRTOffsets_;         //!< Stores the output channel offsets
    std::vector<uint32_t> packOffsets_;   //!< Offset into #weightData_ for a render pass "pack"
    std::vector<uint32_t> packSizes_;     //!< Size (as in number of floats) for each "pack" per render passs
    int markerOffset_;                    //!< Current offset into the packs for strata extraction
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
