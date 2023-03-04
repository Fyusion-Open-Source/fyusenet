//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Transpose Convolutional layer w/ 2x2 mask (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "transconvlayerbase_vanilla.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {

/**
 * @brief Transpose convolution layer for a 2x2 convolution kernel on shallow tensor data
 *
 * This is an implementation of a 2x2 transpose convolution layer, which is usually used for
 * upsampling purposes.
 *
 * @see vanilla::TransConvLayerBase
 */
class TransConvLayer2x2 : public TransConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    TransConvLayer2x2(const ConvLayerBuilder& builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void loadWeightsAndBiases(const float *weights, size_t offset=0) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
};

} // vanilla namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
