//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Avgpool Layer (Header)
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
namespace fyusion::fyusenet::gpu {

/**
 * @brief Average-pooling layer
 *
 * This class implements a 2D average-pooling layer for shallow tensor data. Average pooling
 * computes the average over all values inside a defined window (the "poolsize") along the spatial
 * domain of a tensor. The pooling sizes in this layer are flexible, however we discourage using
 * larger pool sizes than 8x8.
 *
 * @see PoolingLayer
 */
class AvgPoolLayer : public PoolingLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    AvgPoolLayer(const PoolLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void beforeRender() override;
    void afterRender() override;
    void renderChannelBatch(int outPass, int numRenderTargets, int texOffset) override;
    unistateptr initShader(programptr shader, int renderTargets) override;
    programptr compileShader(const char *preproc) override;
};

} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
