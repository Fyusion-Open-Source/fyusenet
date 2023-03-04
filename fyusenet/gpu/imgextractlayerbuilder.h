//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// ImgExtract Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Templatized anchor for GPU-based extract image patches layer
 *
 * @see ImgExtractLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct ImgExtractLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param window Window size for the extract operation (leads to equivalent downsampling)
     *
     * @param name Name for this layer
     */
    ImgExtractLayerBuilderTempl(uint16_t window, const std::string& name) : GPULayerBuilderTempl<D>(name), window_(window) {
      GPULayerBuilderTempl<D>::downsample_[0] = window;
      GPULayerBuilderTempl<D>::downsample_[1] = window;
      LayerBuilderTempl<D>::type_ = LayerType::IMGEXTRACT;
    }

    short window_;
};


/**
 * @brief Builder object for GPU-based extract image patches layer
 *
 * @see DeepExtractImagePatches
 */
struct ImgExtractLayerBuilder : ImgExtractLayerBuilderTempl<ImgExtractLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param window Window size for the extract operation (leads to equivalent downsampling)
     *
     * @param name Name for this layer
     */
    ImgExtractLayerBuilder(short window,const std::string& name) : ImgExtractLayerBuilderTempl<ImgExtractLayerBuilder>(window,name) {}

};


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
