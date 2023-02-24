//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../base/layerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace cpu {

/**
 * @brief Templatized anchor for convolution layer builders for CPU 2D convolution layers
 *
 * @see ConvLayerBuilder
 */
template<typename D = LayerBuilderTempl<>>
struct ConvLayerBuilderTempl : LayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param kernel Spatial kernel size (isotropic, 2D) to use for the convolution
     * @param name Name to be assigned to the layer when built
     */
    ConvLayerBuilderTempl(short kernel, const std::string& name) : LayerBuilderTempl<D>(name),kernel_(kernel) {
    }

    /**
     * @brief Provide isotropic dilation factor for a dilated convolution
     *
     * @param dilate Dilation factor (same for x and y dimension)
     *
     * @return Reference to builder object
     */
    D & dilation(short dilate) {
      dilation_[0] = dilate;
      dilation_[1] = dilate;
      return *(D *)this;
    }

    /**
     * @brief Provide anisotropic dilation factor for a dilated convolution
     *
     * @param horizontal Horizontal dilation factor
     * @param vertical Vertical dilation factor
     *
     * @return Reference to builder object
     */
    D & dilation(short horizontal,short vertical) {
      dilation_[0] = horizontal;
      dilation_[1] = vertical;
      return *(D *)this;
    }

    /**
     * @brief Provide isotropic fractional step for fractional convolutions
     *
     * @param step Fractional step value for fractional convolutions, a value of 0.5 will
     *             perform convolution with a distance of 0.5 between the samples.
     *
     * @return Reference to builder object
     */
    D & sourceStep(float step) {
      sourceStep_ = step;
      return *(D *)this;
    }

    /**
     * @brief Set the group size for grouped convolutions
     *
     * @param gs Group size to set
     *
     * @return Reference to builder object
     */
    D & groupSize(short gs) {
      groupSize_ = gs;
      return *(D *)this;
    }

    short kernel_ = 1;              //!< Isotropic 2D convolution kernel size (we currently do not support anisotropic convolution)
    short dilation_[2] = {1,1};     //!< Dilation factor for dilated convolutions along x- and y-axis
    short groupSize_ = 1;           //!< Group size for grouped/depthwise convolutions (we only support a limited set here)
    float sourceStep_ = 1.f;        //!< Step-size for fractional convolutions
};


/**
 * @brief Builder class for convolution-type layers for CPU 2D convolution layers
 *
 * This class represents a builder for convolution-type layers and adds the specific parameters
 * for those to the general LayerBuilder class. Additional parameters for convolution include:
 *  - kernel size
 *  - dilation factors
 *  - group size
 *  - fractional step values for fractional convolutions
 */
struct ConvLayerBuilder : ConvLayerBuilderTempl<ConvLayerBuilder> {

    ConvLayerBuilder(short kernel, const std::string& name) : ConvLayerBuilderTempl<ConvLayerBuilder>(kernel, name) {}

};

} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
