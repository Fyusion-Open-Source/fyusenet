//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gfxcontextlink.h"
#include "convlayerbuilder.h"
#include "../base/bufferspec.h"
#include "gpulayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu {

/**
 * @brief Base class for (shallow) 2D convolution layers running on the GPU
 *
 * This class serves as base/interface for all GPU-based convolution layers. It contains a very
 * basic standard implementation for basic builder parsing and provides the default interface for
 * cleanup of GPU resources.
 *
 * Further specialization of convolutions is done in the respective base classes for the individual
 * GPU types.
 *
 * @see vanilla::ConvLayerBase
 */
class ConvLayerBase : public GPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ConvLayerBase(const ConvLayerBuilder & builder, int layerNumber);
    ConvLayerBase(const GPULayerBuilder & builder, int layerNumber);
    ~ConvLayerBase() override = default;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
    void loadParameters(const ParameterProvider * weights) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Compile and link required shaders for this layer
     *
     * @post All shaders are compiled and linked
     *
     * @throws GLException if there was an issue with the shader compilation
     */
    virtual void setupShaders() = 0;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int kernel_ = 0;                //!< Kernel size, we currently only support isotropic kernels
    int downsample_[2] = {1,1};     //!< Downsampling per spatial dimension (1 = no downsampling)
    int dilation_[2] = {1,1};       //!< Dilation per spatial dimension (for a trous convolutions), 1 means no dilation / use the direct neighbor
};

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
