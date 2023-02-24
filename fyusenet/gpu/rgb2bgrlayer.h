//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// RGB -> BGR Conversion Layer (Header)
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
#include "functionlayer.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Simple RGB to BGR conversion layer
 *
 * This layer converts an input RGB image to BGR format by a simple swizzlig operation
 */
class RGB2BGRLayer : public FunctionLayer {
 public:
   // ------------------------------------------------------------------------
   // Constructor / Destructor
   // ------------------------------------------------------------------------
   RGB2BGRLayer(const GPULayerBuilder & builder,int layerNumber);

   // ------------------------------------------------------------------------
   // Public methods
   // ------------------------------------------------------------------------
   virtual void cleanup() override;

protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void renderChannelBatch(int outPass,int numRenderTargets,int texOffset) override;
    virtual void setupShaders() override;
    virtual void beforeRender() override;
    virtual void afterRender() override;
    programptr compileShader(const char *preproc);
    unistateptr initShader(programptr shader,int renderTargets);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];        //!< Shader instances (shared) pointers (different shaders for different number of render targets)
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];  //!< Shader states that memorize the shader states of the #shaders_
    ShaderProgram *currentShader_ = nullptr;          //!< Raw pointer to currently active/in-use shader
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
