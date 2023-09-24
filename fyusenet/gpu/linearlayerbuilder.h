//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Linear Layer Builder (Header)                                               (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gfxcontextlink.h"
#include "gpulayerbase.h"
#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
using namespace opengl;
namespace fyusenet::gpu {


/**
 * @brief Templatized anchor for linear layer builders
 *
 * @see LinearLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct LinearLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit LinearLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        GPULayerBuilderTempl<D>::inputChannels_ = LayerBase::PIXEL_PACKING;
        GPULayerBuilderTempl<D>::outputChannels_ = LayerBase::PIXEL_PACKING;
        this->type(LayerType::LINEAR);
    }

    /**
     * @brief Copy-constructor
     *
     * @param src Source builder to copy data from
     */
    LinearLayerBuilderTempl(const LinearLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    /**
     * @brief Set quantization type for this layer
     *
     * @param qType Quantization type
     * @param wtype Data type of quantization weights on the CPU
     *
     * @return Reference to this builder
     */
    D & quantize(qt_type qType, param_type wtype) {
        if (qType != qt_type::QT_MIXED_FLOAT) {
            THROW_EXCEPTION_ARGS(FynException,"Linear layers only support mixed float quantization");
        }
        quantType_ = qType;
        wgtType_ = wtype;
        return *(D *)this;
    }

    /**
     * @brief For quantized layers using GTPQ quantization, defines the quantization group size
     *
     * @param groupSize Quantization group size
     *
     * @return Reference to this builder
     */
    D & quantGroupSize(int groupSize) {
        quantGroupSize_ = groupSize;
        return *(D *)this;
    }

    /**
     * @brief Enable bias / affine mapping for this layer
     *
     * @return Reference to this builder
     */
    D & bias() {
        hasBias_ = true;
        return *(D *)this;
    }

    int quantGroupSize_ = 0;                        //!< For quantized layers using GTPQ quantization, defines the quantization group size
    qt_type quantType_ = qt_type::QT_NONE;          //!< Quantization type
    param_type wgtType_ = param_type::WGT_FLOAT;    //!< Data type of the weights on the CPU
    bool hasBias_ = false;                          //!< Indicator if the layer does an affine mapping
};


/**
 * @brief Builder class for linear layers
 *
 * Linear layers are basically equivalent to matrix multiplications computing a linear or affine
 * mapping on an input tensor.
 */
 // TODO (mw) more documentation
struct LinearLayerBuilder : LinearLayerBuilderTempl<LinearLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit LinearLayerBuilder(const std::string& name) : LinearLayerBuilderTempl<LinearLayerBuilder>(name) {}
};

} // fyusenet::gpu namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
