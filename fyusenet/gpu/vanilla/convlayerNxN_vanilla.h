//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolutional Layer w/ NxN mask (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <mutex>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../gfxcontextlink.h"
#include "../../base/bufferspec.h"
#include "../uniformweightarray.h"
#include "convlayerbase_vanilla.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {

/**
 * @brief Convolution layer using odd NxN convolution kernels for shallow tensors on GPU
 *
 * This class implements a shallow-tensor 2D convolution as laid out in vanilla::ConvLayerBase
 * for odd kernel sizes of size 3x3 and larger running on the GPU.
 *
 * @see vanilla::ConvLayerBase
 */
class ConvLayerNxN : public ConvLayerBase {
 public:
    enum {
        VEC_OVERHEAD = 2,
        RESIDUAL_START_UNIT = GL_TEXTURE1   //!< First texture unit to be used for residual textures
    };
    enum {
        RESIDUAL_SWITCH = 1,
        COEFFICIENTS,
        BIAS,
        BATCHNORM_DATA
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ConvLayerNxN(const ConvLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual void forward(uint64_t sequence = 0) override;
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
