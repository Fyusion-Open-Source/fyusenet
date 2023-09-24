//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Isolated/Explicit SiLU Layer (Header)                                       (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/uniformstate.h"
#include "../gl/fbo.h"
#include "../gl/shaderprogram.h"
#include "gfxcontextlink.h"
#include "sigmoidlayer.h"
#include "../base/layerflags.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu {

/**
 * @brief Layer that maps input data with a SiLU function for shallow & sequence tensors
 *
 * This layer maps all input data element-wise with the SiLU activation function, which is defined
 * as:
 *
 *  \f[ \textnormal{SiLU}(x) = x\left( \frac{1}{1 + \exp^{-x}} \right) \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class SiLULayer : public SigmoidLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    SiLULayer(const GPULayerBuilder & builder, int layerNumber);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;
};

} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
