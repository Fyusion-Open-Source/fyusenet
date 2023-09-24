//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Isolated/Explicit GeLU Layer (Header)                                       (c) Martin Wawro 2023
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
 * @brief Layer that maps input data with the GeLU activation function for shallow & sequence tensors
 *
 * This layer maps all input data element-wise using the GeLU function, that is defined as:
 *
 *  \f[ \textnormal{GeLU}(x) = \frac{1}{2}x \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} ( x + 0.044715x^3) \right)  \right) \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class GeLULayer : public SigmoidLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    GeLULayer(const GPULayerBuilder & builder, int layerNumber);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;
};

} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
