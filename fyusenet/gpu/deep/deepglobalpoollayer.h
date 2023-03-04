//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Global Pooling Layer (Header)
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
 * @brief Global pooling layer for tensors with high channel count (deep format)
 *
 * This layer implements a global pooling operation on the spatial part of the input tensor.
 * Currently two types of pooling operations are supported:
 *   - max-pooling
 *   - average-pooling
 *
 * The output of this layer is a 1x1xC (C being the channel count of the input tensor) tensor.
 */
class DeepGlobalPoolLayer : public DeepPoolingLayer {
 public:
    enum opmode : uint8_t {
        MAXPOOL = 0,
        AVGPOOL
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepGlobalPoolLayer(const PoolLayerBuilder & builder, int layerNumber);

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
    virtual void setupNetworkPolygons(VAO *vao) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the pooling
    unistateptr shaderState_;     //!< UniformState object for the #shader_
    opmode mode_ = MAXPOOL;       //!< Operation mode for this pooling layer (s
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
