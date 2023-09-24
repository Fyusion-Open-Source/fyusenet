//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Dedicated Sigmoid Activation Layer (Header)
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
#include "functionlayer.h"
#include "../base/layerflags.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu {

/**
 * @brief Layer that maps input data with a sigmoid function for shallow & sequence tensors
 *
 * This layer maps all input data element-wise using a sigmoid function, using the following
 * mapping:
 *
 *  \f[ f(x) = \frac{1}{1+e^{-x}} \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class SigmoidLayer : public FunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    SigmoidLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void renderChannelBatch(int outPass,int numRenderTargets,int texOffset) override;
    void setupShaders() override;
    void beforeRender() override;
    void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];          //!< Shader instance (shared) pointers (different shaders for different number of render targets)
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];    //!< Shader states that memorize the shader states of the #shaders_
    ShaderProgram *currentShader_ = nullptr;            //!< Raw pointer to currently active/in-use shader
};

} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
