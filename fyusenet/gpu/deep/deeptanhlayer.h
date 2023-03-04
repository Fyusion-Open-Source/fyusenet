//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep tanh Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../base/bufferspec.h"
#include "deepsigmoidlayer.h"
#include "../scalelayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Layer that maps input data with a sigmoid (tanh) function for deep tensors
 *
 * This layer maps all input data element-wise using the tanh function, using the following
 * mapping:
 *
 *  \f[ \textnormal{tanh}(x) = 2\frac{e^{2x}{1 + e^{2x}} - 1 \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class DeepTanhLayer : public DeepSigmoidLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepTanhLayer(const GPULayerBuilder & builder,int layerNumber);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
