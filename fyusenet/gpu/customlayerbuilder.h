//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Custom GPU Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {

class GPULayerBase;


/**
 * @brief Templatized base for custom layer builder(s) on GPU
 */
template<typename D = GPULayerBuilderTempl<>>
struct CustomLayerBuilderTempl : GPULayerBuilderTempl<D> {
    CustomLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        LayerBuilderTempl<D>::device_ = compute_device::DEV_GPU;
    }
    CustomLayerBuilderTempl(const D& src) : GPULayerBuilderTempl<D>(src) {
    }
};


/**
 * @brief Builder class for custom layers running on GPU
 *
 * This class may be used to build custom layers that are supplied externally. In contrast to the
 * standard builders, it uses a callback function which is to be supplied by the user. This
 * callback must create a new instance derived from GPULayerBase and return it, to be used
 * as result of the build process.
 */
struct CustomLayerBuilder : CustomLayerBuilderTempl<CustomLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     *
     * @param initCB Callback function that is called when the layer for this builder should be
     *               initialized
     */
    CustomLayerBuilder(const std::string& name, std::function<GPULayerBase *(const CustomLayerBuilder & builder)> initCB) : CustomLayerBuilderTempl<CustomLayerBuilder>(name),initCallback_(initCB) {
    }

    /**
     * @brief Copy constructor
     *
     * @param src Object to copy data from
     */
    CustomLayerBuilder(const CustomLayerBuilder & src) : CustomLayerBuilderTempl<CustomLayerBuilder>(src) {}


    /**
     * @brief Initialization function called by the layer facctory
     *
     * @return Pointer to GPU layer that was created by the callback
     *
     * This function diverts program flow to the callback function which was supplied in the
     * constructor and uses the returned GPU layer as result of the build process.
     */
    GPULayerBase * init() const {
        return initCallback_(*this);
    }
 protected:
    /**
     * Callback function that is called upon initialization for the layer.
     */
    std::function<GPULayerBase *(const CustomLayerBuilder & builder)> initCallback_;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
