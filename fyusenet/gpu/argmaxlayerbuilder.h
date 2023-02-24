//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// ArgMax GPU Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gfxcontextlink.h"
#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
using namespace opengl;
namespace fyusenet {
namespace gpu {

/**
 * @brief Templatized anchor for GPU-based arg-max layers
 *
 * @see ArgMaxLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct ArgMaxLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    ArgMaxLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        LayerBuilderTempl<D>::type_ = LayerType::ARGMAX;
    }
};


/**
 * @brief Builder object for GPU-based argmax layers
 *
 * This builder parameterizes the information required to create argmax layers on the GPU. For
 * FyuseNet, argmax layers are somewhat hard due to the restriction that we have to run in fragment
 * shaders. To overcome the associated problems, the GPU-side argmax operation done in FyuseNet is
 * an approximation to the actual argmax function and it should not be used for important
 * classification tasks. It can be used for pixel-based classification if the result does not need
 * to be pixel accurate (for example in a segmentation task).
 *
 * Using this layer in a pure classifier network to determine the actual class of a data-item is
 * discouraged and a CPU-based implementation should be used for that. Please consult the class
 * documentation of the respective layer(s) for more detail.
 *
 * @see DeepArgMaxLayer
 */
struct ArgMaxLayerBuilder : ArgMaxLayerBuilderTempl<ArgMaxLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    ArgMaxLayerBuilder(const std::string& name) : ArgMaxLayerBuilderTempl<ArgMaxLayerBuilder>(name) {        
    }
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
