//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Reduce Layer Builder (Header)
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
 * @brief Templatized anchor for reduction/norm type layer-builders on CPU
 *
 * @see ReduceLayerBuilder
 */
template<typename D = LayerBuilderTempl<>>
struct ReduceLayerBuilderTempl : LayerBuilderTempl<D> {

    /**
     * @brief Enumerator for norm-types
     */
    enum norm {
      NORM_L1,          //!< L1 norm (abs)
      NORM_L2           //!< L2 norm (quadratic norm w/ square root)
    };

    /**
     * @brief Constructor
     *
     * @param redNorm Type of norm to use for reduction (currently we support L1 and L2 norm)
     *
     * @param name Name to be assigned to the layer when built
     */
    ReduceLayerBuilderTempl(norm redNorm,const std::string& name) : LayerBuilderTempl<D>(name),norm_(redNorm) {
    }

    norm norm_;             //!< Norm type to use
};


/**
 * @brief Reduction/Norm layer builder for the CPU
 *
 * This builder is used to create L1/L2 reduction layers across the channel dimension of a tensor.
 */
struct ReduceLayerBuilder : ReduceLayerBuilderTempl<ReduceLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param redNorm Type of norm to use for reduction (currently we support L1 and L2 norm)
     *
     * @param name Name to be assigned to the layer when built
     */
    ReduceLayerBuilder(norm redNorm,const std::string& name) : ReduceLayerBuilderTempl<ReduceLayerBuilder>(redNorm,name) {}

};

} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
