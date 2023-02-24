//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "convlayerbase.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
ConvLayerBase::ConvLayerBase(const ConvLayerBuilder & builder,int layerNumber) : GPULayerBase((const GPULayerBuilder &)builder, layerNumber) {
    assert(builder.type_ != LayerType::ILLEGAL);
    assert(builder.downsample_[0] == builder.downsample_[1]);
    assert(builder.kernel_ > 0);
    assert(builder.dilation_[0] == builder.dilation_[1]);
    kernel_ = builder.kernel_;
    dilation_[0] = builder.dilation_[0];
    dilation_[1] = builder.dilation_[1];
    downsample_[0] = builder.downsample_[0];
    downsample_[1] = builder.downsample_[1];
    leakyReLU_ = builder.leakyReLU_;
    viewport_[0] = (width_ / downsample_[0]) + 2*outputPadding_;
    viewport_[1] = (height_ / downsample_[1]) + 2*outputPadding_;
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
ConvLayerBase::~ConvLayerBase() {
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void ConvLayerBase::cleanup() {
    GPULayerBase::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
