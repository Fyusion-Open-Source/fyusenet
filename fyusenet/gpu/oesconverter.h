//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OES Texture Conversion Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpu/gfxcontextlink.h"
#include "../gl/uniformstate.h"
#include "functionlayer.h"
#include "../base/bufferspec.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

#ifdef FYUSENET_USE_EGL

/**
 * @brief Conversion layer from external OES texture to a plain GL texture
 *
 * This layer simply renders the input OES texture into a plain GL texture. External OES
 * textures are for example used to back SurfaceTexture objects on Android and as such might
 * use a non-RGB input format which is opaquely handled by a special sampler type.
 */
class OESConverter : public FunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    OESConverter(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void beforeRender() override;
    virtual void setupShaders() override;
    virtual void afterRender() override;
    virtual void renderChannelBatch(int outPass,int numRenderTargets,int texOffset) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;
    unistateptr shaderState_;
};

#endif

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
