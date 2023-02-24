//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Fractional Convolutional Layer w/ 3x3 mask (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../gfxcontextlink.h"
#include "../../base/bufferspec.h"
#include "../uniformweightarray.h"
#include "convlayerNxN_vanilla.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {

/**
 * @brief Fractional convolution layer for varying kernel sizes on shallow tensor data
 *
 * This class executes "fractional" convolutions on shallow tensor data using the GPU. Fractional
 * convolutions are basically the same as standard convolutions, but instead of using integer
 * strides, a \e fractional stride is used. This is equivalent of first performing an upsampling
 * operation followed by a standard unit-stride convolution.
 */
class FractionalConvLayerNxN : public ConvLayerNxN {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    FractionalConvLayerNxN(const ConvLayerBuilder & builder, int layerNumber);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void compileConvolutionShaders(const char *preproc) override;
};

} // vanilla namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
