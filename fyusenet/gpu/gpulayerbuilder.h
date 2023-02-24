//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Generic GPU Layer Builder (Header)
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
#include "../base/layerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
using namespace opengl;
namespace fyusenet {
namespace gpu {

/**
 * @brief Templatized anchor for GPU-based layer builders
 */
template<typename D = LayerBuilderTempl<>>
struct GPULayerBuilderTempl : LayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    GPULayerBuilderTempl(const std::string& name) : LayerBuilderTempl<D>(name) {
      LayerBuilderTempl<D>::device_ = compute_device::DEV_GPU;
    }

    /**
     * @brief Copy constructor
     *
     * @param src Builder object to copy
     */
    GPULayerBuilderTempl(const D& src) : LayerBuilderTempl<D>(src) {
      context_ = src.context_;
    }

    /**
     * @brief Set GL context (link) for the layer
     *
     * @param context OpenGL context link to be used for the newly built layer
     *
     * @return Reference to builder object
     */
    D & context(GfxContextLink context) {
      context_ = context;
      return *(D *)this;
    }

    GfxContextLink context_;                     //!< GL context to use for the newly-built layer
};

/**
 * @brief Base-class for GPU-based layer builders
 */
struct GPULayerBuilder : GPULayerBuilderTempl<GPULayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    GPULayerBuilder(const std::string& name) : GPULayerBuilderTempl<GPULayerBuilder>(name) {
    }

    /**
     * @brief Copy constructor
     *
     * @param src Builder object to copy
     */
    GPULayerBuilder(const GPULayerBuilder & src) : GPULayerBuilderTempl<GPULayerBuilder>(src) {}
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
