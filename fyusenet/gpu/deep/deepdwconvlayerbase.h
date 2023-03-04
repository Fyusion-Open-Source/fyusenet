//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Depth-Wise Convolution Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../base/bufferspec.h"
#include "../convlayerbase.h"
#include "deeptiler.h"
#include "deepconvlayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Base class for deep-tensor depth-wise convolution layers (width/height <= 48 and depth >= 96)
 *
 * This class contains some base implementation for depthwise convolution layers operating on
 * deep-channel tensors. Depthwise convolution layers are different from normal convolution layers
 * because they are not computing the inner product along the input channel axis for every output
 * channel. Instead, the number of input and output channels are equivalent
 */
class DeepDepthwiseConvLayerBase : public DeepConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepDepthwiseConvLayerBase(const ConvLayerBuilder& builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void loadWeightsAndBiases(const float *weights,size_t offset=0) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void createWeightTextureMatrix(const float *srcweights, int winOffset, GLuint weightTexture);
    virtual void setupNetworkPolygons(VAO *vao) override;
    virtual void setupShaders() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int channelMultiplier_ = 1;     //!<

    // ------------------------------------------------------------------------
    // Internal constants
    // ------------------------------------------------------------------------
    constexpr const static int WEIGHT_TEXTURE = 4;   //!< Texture unit for weights
    constexpr const static int BIAS_TEXTURE = 5;     //!< Texture unit for biases
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
