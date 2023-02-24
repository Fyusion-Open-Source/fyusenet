//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolutional Layer w/ 1x1 kernel (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

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

#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../gfxcontextlink.h"
#include "../../base/bufferspec.h"
#include "../uniformweightarray.h"
#include "../convweightarrayKxKxNxM.h"
#include "convlayerbase_vanilla.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {

/**
 * @brief Convolution layer using 1x1 convolution kernels for shallow tensors on GPU
 *
 * This class implements a shallow-tensor 2D convolution as laid out in vanilla::ConvLayerBase
 * for a 1x1 kernel running on the GPU. Technically, this is equivalent to a fully-connected layer.
 *
 * @see vanilla::ConvLayerBase
 */
class ConvLayer1x1 : public ConvLayerBase {
 public:
    constexpr static int CONVSIZE = 1;                  //!< Convolution kernel size
    constexpr static int RESIDUAL_START_UNIT = 4;       //!< First available texture unit for routing in residuals

    /** Enumerator for shader symbol IDs */
    enum {
        RESIDUAL_SWITCH,
        COEFFICIENTS,
        BIAS,
        BATCHNORM_DATA,
        INPUT_TEX_TRANSFORM
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ConvLayer1x1(const ConvLayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual void forward(uint64_t sequence) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
    virtual void compileConvolutionShaders(const char *preproc);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::vector<programptr> convolutionShaders_;
    std::vector<unistateptr> convolutionShaderStates_;
};

} // vanilla namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
