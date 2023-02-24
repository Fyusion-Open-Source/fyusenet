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

#ifdef ANDROID
#include <GLES3/gl3.h>
#else
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <OpenGL/glext.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#endif

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/uniformstate.h"
#include "../gl/fbo.h"
#include "../gl/vao.h"
#include "../gl/vbo.h"
#include "../gl/ibo.h"
#include "../base/bufferspec.h"
#include "functionlayer.h"
#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

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
    virtual void beforeRender() override;
    virtual void afterRender() override;
    virtual void setupShaders() override;
    unistateptr initShader(programptr shader,int renderTargets);
    programptr compileShader(const char *preproc);
    virtual void renderChannelBatch(int outPass, int numRenderTargets, int texOffset) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];
    ShaderProgram *currentShader_ = nullptr;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
