//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Convolutional Layer w/ 1x1 mask (Header)
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
#include "../../base/bufferspec.h"
#include "../uniformweightarray.h"
#include "deepconvlayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief 1x1 convolution layer for deep tensor format
 *
 * This class implements a 1x1 convolution layer for deep tensor formats on GPU as laid out in
 * deep::DeepConvLayerBase in more detail.
 *
 * @see deep::DeepConvLayerNxN
 */
class DeepConvLayer1x1 : public DeepConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepConvLayer1x1(const ConvLayerBuilder & buidler, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void forward(uint64_t sequence) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void compileConvolutionShaders(const char *preproc) override;
    unistateptr initShader(programptr shader);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------

};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
