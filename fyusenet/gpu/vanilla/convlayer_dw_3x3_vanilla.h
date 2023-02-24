//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Depthwise Convolutional Layer w/ 3x3 mask (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../gfxcontextlink.h"
#include "../../base/bufferspec.h"
#include "../uniformweightarray.h"
#include "convlayerNxN_vanilla.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {

/**
 * @brief Depthwise convolution layer for 3x3 convolutions on shallow-format tensors on the GPU
 *
 * This class implements a depthwise convolution with a 3x3 kernel on deep-format tensors. In
 * contrast to "normal" convolutions, depthwise convolutions use a 3D convolution filter and
 * they add the restriction that the number of input channels is equal to the number of output
 * channels.
 *
 * Instead of performing an additional inner product over the input channels. a depthwise
 * convolution computes a convolution on a per-channel basis by only using a single slice of
 * the convolution filter, which corresponds to that channel. Mathematically:
 * \f[ t_o(i,j,k) = \sum_{m,n} t_i_(i-m,j-n,k} \cdot \kappa(m,n,k) \f]
 * where \f$ \kappa \f$ refers to the convolution kernel weights.
 *
 * Depthwise convolution layers are often paired with 1x1 convolutions to form a block that is
 * denoted as "depthwise separable convolution", a technique which has been popularized by
 * "MobileNets".
 *
 * @todo Generalize this to NxN convolutions
 */
class DepthwiseConvLayer3x3 : public ConvLayerNxN {
    // TODO (mw) generalize this to NxN convolutions
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DepthwiseConvLayer3x3(const ConvLayerBuilder& builder, int layerNumber);
    virtual ~DepthwiseConvLayer3x3();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void loadWeightsAndBiases(const float *weights, size_t offset=0) override;
    virtual void forward(uint64_t sequence = 0) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setBias(int outPass, const UniformWeightArray *bias) override;
    virtual void compileConvolutionShaders(const char *preproc) override;
    virtual void setupNetworkPolygons(VAO *vao, int kernel) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr compileSingleShader(int outputDepth, int inputDepth, const char *preproc);
    int channelMultiplier_ = 1;
    int maxInputTextures_ = 1;
};

} // vanilla namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
