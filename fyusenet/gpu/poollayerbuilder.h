//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Pooling Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gpulayerbuilder.h"
#include "../common/fynexception.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Templatized anchor for GPU-based pool layer builder(s)
 */
template<typename D = GPULayerBuilderTempl<>>
struct PoolLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Enumerator for pooling mode
     */
    enum op {
        POOL_AVG = 0,           //!< Average pooling (box filtering)
        POOL_MAX                //!< Max-pooling
    };

    /**
     * @brief PoolLayerBuilderTempl
     *
     * @param poolOp Pool operation to use (either maximum or average pooling)
     *
     * @param name Name to be assigned to the built layer
     */
    PoolLayerBuilderTempl(op poolOp,const std::string& name) : GPULayerBuilderTempl<D>(name),operation_(poolOp) {
        switch (poolOp) {
            case POOL_AVG:
                LayerBuilderTempl<D>::type_ = LayerType::AVGPOOL2D;
                break;
            case POOL_MAX:
                LayerBuilderTempl<D>::type_ = LayerType::MAXPOOL2D;
                break;
            default:
                assert(false);
        }
    }


    /**
     * @brief Set the pooling size (isotropic)
     *
     * @param win Pool size along x- and y-dimension
     *
     * @return Reference to builder object
     *
     * @note The pool size does not automatically control the downsampling factor, see
     *       downsample() for that.
     */
    D & poolSize(short win) {
        poolsize_[0] = win;
        poolsize_[1] = win;
        return *(D *)this;
    }


    /**
     * @brief Set the pooling size (anisotropic)
     *
     * @param winx Pool size along x-dimension
     * @param winy Pool size along y-dimension
     *
     * @return Reference to builder object
     *
     * @note The pool size does not automatically control the downsampling factor, see
     *       downsample() for that.
     */
    D & poolSize(short winx,short winy) {
        poolsize_[0] = winx;
        poolsize_[1] = winy;
        return *(D *)this;
    }


    /**
     * @brief Set the global pooling flag
     *
     * @return Reference to builder object
     *
     * When global pooling is turned on, the data is spatially pooled to a 1x1 width/height
     * dimension without changing the number of channels.
     */
    D & global() {
        if ((LayerBuilderTempl<D>::width_==0) || (LayerBuilderTempl<D>::height_==0)) THROW_EXCEPTION_ARGS(FynException,"Must set size before specifying global pooling");
        LayerBuilderTempl<D>::downsample_[0]=LayerBuilderTempl<D>::width_;
        LayerBuilderTempl<D>::downsample_[1]=LayerBuilderTempl<D>::height_;
        global_ = true;
        poolsize_[0] = LayerBuilderTempl<D>::width_;
        poolsize_[1] = LayerBuilderTempl<D>::height_;
        return *(D *)this;
    }

    op operation_;                  //!< Pooling operation to be used (avg or max)
    short poolsize_[2] = {1,1};     //!< Pooling size along x- and y-dimension
    bool global_ = false;           //!< Flag that enables global pooling
};


/**
 * @brief Builder class for pooling layers running on the GPU
 *
 * This class encapsulates the parameter for building a pooling layer. It exposes an interface
 * to adjust the pooling type, which can either be average-pooling or max-pooling, as well as
 * the possibility to set the pooling size. The pooling size refers to the number of spatially
 * neighboring pixels that are to be combined using the selected operation.
 *
 * The downsampling for the pooling is not directly controlled by the pooling size, but by the
 * #downsample() call.
 */
struct PoolLayerBuilder : PoolLayerBuilderTempl<PoolLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param poolOp Pool operation to use (either maximum or average pooling)
     *
     * @param name Name to be assigned to the built layer
     */
    PoolLayerBuilder(op poolOp,const std::string& name) : PoolLayerBuilderTempl<PoolLayerBuilder>(poolOp,name) {}
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
