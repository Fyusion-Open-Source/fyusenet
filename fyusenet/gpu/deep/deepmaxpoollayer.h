//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Maxpool Layer (Header)
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

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Max-pooling layer for deep tensor data
 *
 * This class implement a 2D max-pooling layer for deep tensor data. Maximum pooling simply
 * computes the maximum over all values inside a defined window along the spatial domain of a
 * tensor. The pooling sizes in this layer are flexible, however using pool sizes larger than 8x8
 * is discouraged.
 *
 * @see DeepPoolingLayer
 */
class DeepMaxPoolLayer : public DeepPoolingLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepMaxPoolLayer(const PoolLayerBuilder &builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void renderChannelBatch() override;
    virtual void beforeRender() override;
    virtual void setupShaders() override;
    virtual void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the pooling
    unistateptr shaderState_;     //!< UniformState object for the #shader_
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:


