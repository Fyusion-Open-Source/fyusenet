//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Cast Layer Builder (Header)
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
 * @brief Templatized anchor for type-cast emulation layers on the GPU
 *
 * @see CastLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct CastLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     * @param tgt Target datatype to cast the tensor data to
     *
     * @see CastTarget
     */
    CastLayerBuilderTempl(const std::string& name, CastTarget tgt) : GPULayerBuilderTempl<D>(name),target_(tgt) {
        LayerBuilderTempl<D>::type_ = LayerType::CAST;
    }

    /**
     * @brief Copy-constructor
     *
     * @param src Source builder to copy data from
     */
    CastLayerBuilderTempl(const CastLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    /**
     * @brief Set target datatype to cast the tensor data to
     *
     * @param tgt Target datatype to cast the tensor data to
     *
     * @return Reference to builder object
     *
     * @todo Maybe remove this, because the target type is already supplied in the constructor and
     *       an override like this is most likely not necessary.
     *
     *  @see CastTarget
     */
    D & target(CastTarget tgt) {
        target_ = tgt;
        return *this;
    }

    CastTarget target_;             //!< Tarrget datatype to cast tensor data to
};


/**
 * @brief Builder class for type-cast emulation layers on GPU
 *
 * This class serves as the parameter collector for building type-cast emulation layers.
 */
struct CastLayerBuilder : CastLayerBuilderTempl<CastLayerBuilder> {
    CastLayerBuilder(const std::string& name, CastTarget tgt) : CastLayerBuilderTempl<CastLayerBuilder>(name, tgt) {}
    using CastLayerBuilderTempl<CastLayerBuilder>::CastLayerBuilderTempl;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
