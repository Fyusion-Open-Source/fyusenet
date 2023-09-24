//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Sigmoid Layer (Header)
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
#include "deepfunctionlayer.h"
#include "../scalelayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::deep {


/**
 * @brief Layer that maps input data with a sigmoid function for deep tensors
 *
 * This layer maps all input data element-wise using a sigmoid function, using the following
 * mapping:
 *
 *  \f[ f(x) = \frac{1}{1+e^{-x}} \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class DeepSigmoidLayer : public DeepFunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepSigmoidLayer(const GPULayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;
    void renderChannelBatch() override;
    void beforeRender() override;
    void afterRender() override;
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the pooling
    unistateptr shaderState_;     //!< UniformState object for the #shader_
};

} // fyusion::fyusenet::gpu::deep namespace


// vim: set expandtab ts=4 sw=4:
