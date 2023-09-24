//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Reduce Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "cpulayerbase.h"
#include "reducelayerbuilder.h"

namespace fyusion::fyusenet::cpu {
//------------------------------------- Public Declarations ----------------------------------------


/**
 * @brief Reduction layer (CPU-based)
 *
 * This layer performs a reduction operation by calculating either the L1 or the L2 norm of an
 * input tensor across its channels (\e not in the spatial domain) and outputs a single-channel
 * tensor as a result.
 *
 * @warning The code in this class is not optimized at all and may run rather slowly. This layer
 *          is meant to be used on the trailing end of a network when most of the data processing
 *          has already been done on the GPU, hopefully resulting in rather small resulting tensors.
 *
 * @todo Come up with an optimized/vectorized version of this layer
 */
class ReduceLayer : public CPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ReduceLayer(const ReduceLayerBuilder& builder, int layerNumber);
    ~ReduceLayer() override = default;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    void forward(uint64_t sequenceNo, StateToken * state) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void reduceL1AcrossChannels(const float *input, float *output);
    void reduceL2AcrossChannels(const float *input, float *output);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    ReduceLayerBuilder::norm norm_;   //!< Type of norm to use for reduction, either L1 or L2 norm are currently supported
};



} // fyusion::fyusenet::cpu namespace

// vim: set expandtab ts=4 sw=4:
