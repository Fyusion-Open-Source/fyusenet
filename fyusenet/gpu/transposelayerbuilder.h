//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Spatial Transposition GPU Layer Builder (Header)
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
 * @brief Templatized anchor for transposition layer
 */
template<typename D = GPULayerBuilderTempl<>>
struct TransposeLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief TransposeLayerBuilderTempl
     * @param name
     */
    TransposeLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        LayerBuilderTempl<D>::type_ = LayerType::TRANSPOSE;
    }

    TransposeLayerBuilderTempl(const TransposeLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    bool equal() const {
        return true;
    }
};


/**
 * @brief Builder object for transposition layers
 *
 *
 *
 * @see DeepTransposeLayer
 */
struct TransposeLayerBuilder : TransposeLayerBuilderTempl<TransposeLayerBuilder> {
    TransposeLayerBuilder(const std::string& name) : TransposeLayerBuilderTempl<TransposeLayerBuilder>(name) {}
    using TransposeLayerBuilderTempl<TransposeLayerBuilder>::TransposeLayerBuilderTempl;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
