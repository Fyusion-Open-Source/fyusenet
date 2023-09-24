//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "convlayerbase.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
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
    viewport_[0] = (width_ / downsample_[0]) + 2*outputPadding_;
    viewport_[1] = (height_ / downsample_[1]) + 2*outputPadding_;
    hasParameters_ = true;
}


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
ConvLayerBase::ConvLayerBase(const GPULayerBuilder & builder,int layerNumber) : GPULayerBase(builder, layerNumber) {
    assert(builder.type_ != LayerType::ILLEGAL);
    kernel_ = 1;
    viewport_[0] = width_ + 2*outputPadding_;
    viewport_[1] = height_ + 2*outputPadding_;
    hasParameters_ = true;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void ConvLayerBase::cleanup() {
    GPULayerBase::cleanup();
}


/**
 * @brief Load weights, biases and more from parameter provider
 *
 * @param weights Pointer to ParameterProvider instance that supplies all (constant) data
 *
 * This function extracts weights, biases and also batch-norm parameters stored in the
 * \p biasAndWeights source. For convolutions, the convolution weights are supposed to be
 * stored in the following nested order:
 *
 * @code
 * [outchannel][kernely][kernelx][inchannel]
 * @endcode
 *
 * Thus for \e m input channels and a kernel of size \e k (i.e. a \f$ k \times k \f$ kernel),
 * we expect a 4D array of size \f$ n \times k \times k \times m \f$. The bias data as well
 * is supposed to be stored as simple 1D vectors and the batch-norm parameters are expected
 * to be stored in the following order:
 *  1. all scales (single value per output channel for a total of \c output values)
 *  2. all offsets (single value per output channel for a total of \c output values)
 *
 * The parameter provider is called with the following \c name parameters on loading data:
 *   - \c layername.weights for the weight data, \c subIndex set to 0
 *   - \c layername.bias for the bias data, \c subIndex set to 1
 *   - \c layername.bn for the batch-norm data, \c subIndex set to 2
 *
 * Where \c layername is the name that was assigned to the layer by the builder.
 *
 * @see ConvWeightArrayKxKxNxM, ParameterProvider
 *
 * @note It is safe to call this function from a context that is shared with the initial GL
 *       context that was used to create the layer.
 */
void ConvLayerBase::loadParameters(const fyusion::fyusenet::ParameterProvider *weights) {
    (void)weights;
    // NOTE (mw) this function is only here for the documentation
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
