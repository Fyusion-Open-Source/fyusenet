//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Maxpool Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/shaderprogram.h"
#include "../gl/uniformstate.h"
#include "poolinglayer.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Max-pooling layer for shallwo tensor data
 *
 * This class implement a 2D max-pooling layer for shallow tensor data. Maximum pooling simply
 * computes the maximum over all values inside a defined window along the spatial domain of a
 * tensor. The pooling sizes in this layer are flexible, however using pool sizes larger than 8x8
 * is discouraged,
 *
 * @see PoolingLayer
 */
class MaxPoolLayer : public PoolingLayer {
 public:
   // ------------------------------------------------------------------------
   // Constructor / Destructor
   // ------------------------------------------------------------------------
   MaxPoolLayer(const PoolLayerBuilder & builder, int layerNumber);

   // ------------------------------------------------------------------------
   // Public methods
   // ------------------------------------------------------------------------

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void beforeRender() override;
    virtual void afterRender() override;
    virtual void renderChannelBatch(int outPass,int numRenderTargets,int texOffset) override;
    virtual unistateptr initShader(programptr shader,int renderTargets) override;
    virtual programptr compileShader(const char *preproc) override;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
