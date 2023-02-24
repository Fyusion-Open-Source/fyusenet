//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Scaling Layer for Shallow Tensors (Header)
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
#include "../base/modifierinterfaces.h"
#include "functionlayer.h"
#include "scalelayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Spatial scaling (up/down) for shallow tensor data
 *
 * This layer implements upscaling and downscaling (by integer factors) for shallow tensor data.
 * Scaling can either be done using linear interpolation or nearest-neighbor interpolation. As
 * this layer is one of the least complex one (if not \e the least complex), it can also be used
 * to add/remove padding from tensors by performing an in-GPU copy.
 */
class ScaleLayer : public FunctionLayer, public RotationModifier {
    enum {
        TEXTRANS = 1
    };
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ScaleLayer(const ScaleLayerBuilder & builder, int layerNumber);
    ScaleLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void setRotation(int degrees) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
    virtual void beforeRender() override;
    virtual void afterRender() override;
    virtual void renderChannelBatch(int outPass, int numRenderTargets, int texOffset) override;
    unistateptr initShader(programptr shader, int renderTargets);
    programptr compileShader(const char *preproc);
    void rotate(int degrees);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];
    ShaderProgram *currentShader_ = nullptr;
    ScalingType type_ = ScalingType::NEAREST;
    int rotation_ = 0;
    GLfloat textureMatrix_[16] = {0};
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
