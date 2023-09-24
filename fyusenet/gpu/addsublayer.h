//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Explicit Add/Sub Layer (Header)
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
#include "../base/layerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu {

/**
 * @brief Simple addition / subtraction layer for shallow tensors
 *
 * This layer implements a simple channel-wise addition (or subtraction) operation of two tensors.
 * In case of subtraction, this layer subtracts the data supplied in port 1 from the data supplied
 * in port 0.
 *
 * @note In most cases it is better/faster to use the layerflags::RESIDUAL_INPUT flag to add two
 *       tensors by fusing the operation with another operation.
 */
class AddSubLayer : public FunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    AddSubLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
    std::vector<BufferSpec> getRequiredInputBuffers() const override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void renderChannelBatch(int outPass,int numRenderTargets,int texOffset) override;
    void setupShaders() override;
    void beforeRender() override;
    void afterRender() override;
    programptr compileShader(const char *preproc);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];          //!< Shader instance (shared) pointers (different shaders for different number of render targets)
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];    //!< Shader states that memorize the shader states of the #shaders_
    ShaderProgram *currentShader_ = nullptr;            //!< Raw pointer to currently active/in-use shader
    bool negative_ = false;                             //!< If set to true, this layer performs subtraction
    mutable int texturesPerPort_ = 0;                   //!< Number of input textures per port
};

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
