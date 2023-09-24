//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// 2D Non-Maximum Suppression Layer
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
#include "../gl/vao.h"
#include "../gl/vbo.h"
#include "../gl/ibo.h"
#include "../base/bufferspec.h"
#include "functionlayer.h"
#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu {

/**
 * @brief Layer that performs non-maximum suppresion on the spatial (2D) part of a tensor in shallow representation
 *
 * This class constitutes a layer that performs a 2D non-maximum-suppression task in a 3x3 neighborhood
 * of the spatial part of a tensor. It is specific to shallow-formatted tensor data.
 */
class NonMaxSuppression2D : public FunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    NonMaxSuppression2D(const GPULayerBuilder &builder, int layerNumber);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void beforeRender() override;
    void afterRender() override;
    void setupShaders() override;
    unistateptr initShader(programptr shader,int renderTargets);
    programptr compileShader(const char *preproc);
    void renderChannelBatch(int outPass, int numRenderTargets, int texOffset) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];
    ShaderProgram *currentShader_ = nullptr;
};

} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
