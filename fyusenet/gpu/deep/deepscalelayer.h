//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Scaling Layer (Header)
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

#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../base/bufferspec.h"
#include "deepfunctionlayer.h"
#include "../scalelayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Spatial scaling layer for deep tensor data
 *
 * This layer implements upscaling and downscaling (by integer factors) for deep-channel tensor data.
 * Scaling can either be done using linear interpolation or nearest-neighbor interpolation. As
 * this layer is one of the least complex one (if not \e the least complex), it can also be used
 * to add/remove padding from tensors by performing an in-GPU copy.
 */
class DeepScaleLayer : public DeepFunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepScaleLayer(const ScaleLayerBuilder & builder,int layerNumber);
    DeepScaleLayer(const GPULayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
    virtual void renderChannelBatch() override;
    virtual void beforeRender() override;
    virtual void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the scaling
    unistateptr shaderState_;     //!< UniformState object for the #shader_
    ScalingType type_;            //!< Scaling type (e.g. nearest neighbor, lerp etc.)
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
