//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Singleton Arithmetic Layer Builder (Header)
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
namespace fyusenet {
namespace gpu {


/**
 * @brief Templatized anchor for layer builders for GPU-based singleton arithmetic layers
 *
 * @see SingletonArithLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct SingletonArithLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name of the layer
     * @param type Operation type for this layer
     */
    SingletonArithLayerBuilderTempl(const std::string& name, ArithType type) : GPULayerBuilderTempl<D>(name), opType_(type) {
        LayerBuilderTempl<D>::type_ = LayerType::SINGLETON_ARITH;
    }

    /**
     * @brief Copy constructor
     *
     * @param src Object to copy from
     */
    SingletonArithLayerBuilderTempl(const SingletonArithLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    /**
     * @brief Define operand value for the operation
     *
     * @param opd Operand value
     *
     * @return Reference to builder object
     */
    D & operand(float opd) {
        operand_ = opd;
        return *(D *)this;
    }

    ArithType opType_;              //!< Operation type
    float operand_ = 0.0f;          //!< Operand (singleton) for the operation
};


/**
 * @brief Builder class for GPU-based singleton arithmetic layers
 *
 * This class represents a builder for simple arithmetic layers that involve a tensor and a single
 * operand/operation. For example, adding a constant number to all elements of a tensor.
 * The following operations are supported:
 *   - addition / subtraction
 *   - multiplication / division
 *
 * @see SingletonArithmeticLayer
 */
struct SingletonArithLayerBuilder : SingletonArithLayerBuilderTempl<SingletonArithLayerBuilder> {
    SingletonArithLayerBuilder(const std::string& name, ArithType type) : SingletonArithLayerBuilderTempl<SingletonArithLayerBuilder>(name, type) {}
    using SingletonArithLayerBuilderTempl<SingletonArithLayerBuilder>::SingletonArithLayerBuilderTempl;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
