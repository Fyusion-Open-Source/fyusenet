//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Convolution Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "cpulayerbase.h"
#include "convlayerbuilder.h"

namespace fyusion::fyusenet::cpu {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Basic implementation for CPU-based convolution layers
 *
 * This class implements basic 2D convolutions on the CPU. As of time of writing, FyuseNet is
 * GPU-centric and CPU-based convolutions only make sense for very small tensors that occur at
 * the beginning or end of a processing pipeline. For this reason, this layer does not contain
 * any type of performance optimization (not even basic one) and also does not offer the same
 * degree of functionality as the GPU-based layers.
 *
 * This is not to say, that the CPU-side of FyuseNet cannot/will not be optimized in the future.
 */
class ConvolutionLayer : public CPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ConvolutionLayer(const ConvLayerBuilder& builder, int layerNumber);
    ~ConvolutionLayer() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void loadParameters(const ParameterProvider *weights) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void preReLU(float *data);
    void postReLU(float *data);
    void paddedConv(const float *input, float *output);
    void unpaddedConv(const float *input, float *output);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int kernel_ = 0;
    int dilation_[2] = {1,1};
    int upsample_[2] = {1,1};
    int downsample_[2] = {1,1};
    float * weights_ = nullptr;
    float * bias_ = nullptr;
    float * bnScale_ = nullptr;
};

} // fyusion::fyusenet::cpu namespace

// vim: set expandtab ts=4 sw=4:
