//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Blur GPU Layer Builder (Header)
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
#include "gpulayerbase.h"
#include "../base/layerbuilder.h"
#include "../base/layerflags.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
using namespace opengl;
namespace fyusenet {
namespace gpu {

/**
 * @brief Templatized anchor for blurring layer builders on the GPU
 *
 * @see BlurLayerBuiler
 */
template<typename D = GPULayerBuilderTempl<>>
struct BlurLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    BlurLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        LayerBuilderTempl<D>::type_ = LayerType::BLUR2D;
    }

    /**
     * @brief Set kernel size for the blur operation
     *
     * @param sz Kernel size (2D isotropic) for the smoothing
     *
     * @return Reference to builder object
     *
     * @note Kernels should be small. Larger kernel sizes (>=7) may lead to slow performance as
     *       the code is not optimized for larger kernels.
     *
     * The default kernel size is 3.
     */
    D & kernel(int sz) {
      kernel_ = sz;
      return *(D *)this;
    }

    /**
     * @brief Set blur type
     *
     * @param typ Blur-type to be applied, either \c AVERAGE (box-filter) or \c GAUSSIAN
     *
     * @return Reference to builder object
     *
     * The default filter type is the \c AVERAGE filter.
     *
     */
    D & blurType(BlurKernelType typ) {
      blurType_ = typ;
      return *(D *)this;
    }

    BlurKernelType blurType_ = BlurKernelType::AVERAGE;     //!< Blur-kernel type
    int kernel_ = 3;                                        //!< Blur kernel size
};


/**
 * @brief Builder class for blurring layers on the GPU
 *
 * This class is to be used to build blur layers running on the GPU.
 */
struct BlurLayerBuilder : BlurLayerBuilderTempl<BlurLayerBuilder> {
    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    BlurLayerBuilder(const std::string & name) : BlurLayerBuilderTempl<BlurLayerBuilder>(name) {}
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
