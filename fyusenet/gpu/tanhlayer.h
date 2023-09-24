//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Isolated/Explicit tanh Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

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
 * @brief Layer that maps input data with a sigmoid (tanh) function for shallow & sequence tensors
 *
 * This layer maps all input data element-wise using the tanh function, using the following
 * mapping:
 *
 *  \f[ \textnormal{tanh}(x) = 2\frac{e^{2x}{1 + e^{2x}} - 1 \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class TanhLayer : public SigmoidLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    TanhLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------

};

} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
