//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Average-Pool Layer (Header)
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
#include "../../gl/shaderprogram.h"
#include "deeppoolinglayer.h"
#include "../poollayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::deep {

/**
 * @brief Average-pooling layer for deep tensor data
 *
 * This class implements a 2D average-pooling layer for deep tensor data. Average pooling
 * computes the average over all values inside a defined window (the "poolsize") along the spatial
 * domain of a tensor. The pooling sizes in this layer are flexible, however we discourage using
 * larger pool sizes than 8x8.
 *
 * @see DeepPoolingLayer
 */
class DeepAvgPoolLayer : public DeepPoolingLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepAvgPoolLayer(const PoolLayerBuilder &builder, int layerNumber);
    void cleanup() override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void renderChannelBatch() override;
    void beforeRender() override;
    void setupShaders() override;
    void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the pooling
    unistateptr shaderState_;     //!< UniformState object for the #shader_
};

} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
