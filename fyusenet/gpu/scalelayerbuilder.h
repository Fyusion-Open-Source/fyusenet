//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Scaling GPU Layer Builder (Header)
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
 * @brief Templatized anchor for spatial scaling-type layers on the GPU
 *
 * @see ScaleLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct ScaleLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    ScaleLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name),scaleType_(ScalingType::NEAREST) {
    }

    /**
     * @brief Copy-constructor
     *
     * @param src Source builder to copy data from
     */
    ScaleLayerBuilderTempl(const ScaleLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src),scaleType_(ScalingType::NEAREST) {
    }

    /**
     * @brief Set scaling type for the layer
     *
     * @param typ Scaling type, (either \c NEAREST or \c LINEAR), default is \c NEAREST
     *
     * @return Reference to builder object
     */
    D & scaleType(ScalingType typ) {
        scaleType_ = typ;
        return *(D *)this;
    }

    /**
     * @brief Set rotation angle (in degrees)
     *
     * @param angle Rotation angle
     *
     * @return Reference to builder object
     *
     * Sets the rotation angle for the scaling layer, which applies a counterclockwise rotation
     * to the input data (when considered as 2D image).
     *
     * @note This functionality is not very-well tested and should be used with caution
     */
    D & rotate(int angle) {
        rotation_ = angle;
        return *(D *)this;
    }

    /**
     * @brief Set isotropic scale factor
     *
     * @param sc Scaling factor, will be the same for the x- and y-dimension
     *
     * @return Reference to builder object
     */
    D & scale(float sc) {
        return scale(sc, sc);
    }

    /**
     * @brief Set anisotropic scale factors
     *
     * @param scaleX Scaling factor along x-dimension
     *
     * @param scaleY Scaling factor along y-dimension
     *
     * @return Reference to builder object
     */
    D & scale(float scaleX, float scaleY) {
        if (scaleX > 1.0f) {
            LayerBuilderTempl<D>::upsample_[0] = (int)scaleX;
            if (fabs(LayerBuilderTempl<D>::upsample_[0] - scaleX) > 1e-4) {
                THROW_EXCEPTION_ARGS(FynException,"Only supporting integer upscales for now");
            }
        }
        if (scaleY > 1.0f) {
            LayerBuilderTempl<D>::upsample_[1]=(int)scaleY;
            if (fabs(LayerBuilderTempl<D>::upsample_[1] - scaleY) > 1e-4) {
                THROW_EXCEPTION_ARGS(FynException,"Only supporting integer upscales for now");
            }
        }
        if (scaleX < 1.0f) {
            float dn = 1.0f/scaleX;
            LayerBuilderTempl<D>::downsample_[0]=(int)dn;
            if (fabs(LayerBuilderTempl<D>::downsample_[0] - dn) > 1e-4) {
                THROW_EXCEPTION_ARGS(FynException,"Only supporting integer downscales for now");
            }
        }
        if (scaleY < 1.0f) {
            float dn = 1.0f/scaleY;
            LayerBuilderTempl<D>::downsample_[0]=(int)dn;
            if (fabs(LayerBuilderTempl<D>::downsample_[1] - dn) > 1e-4) {
                THROW_EXCEPTION_ARGS(FynException,"Only supporting integer upscales for now");
            }
        }
        return *(D *)this;
    }

    /**
     * @brief Check if scaling is isotropic
     *
     * @retval true scaling is isotropic
     * @retval false scaling is not isotropic
     */
    bool equal() const {
        return (LayerBuilderTempl<D>::upsample_[0] == LayerBuilderTempl<D>::upsample_[1]) && (LayerBuilderTempl<D>::downsample_[0] == LayerBuilderTempl<D>::downsample_[1]);
    }

    int rotation_ = 0;                              //!< Rotation angle (in degrees)
    ScalingType scaleType_ = ScalingType::NEAREST;  //!< Scaling interpolation mode (default is \c NEAREST)
};


/**
 * @brief Builder class for scaling type layers on GPU
 *
 * This builder class is to be used for building 2D spatial scaling layers on the GPU. Aside from
 * scaling, this layer also supports rotation, which should be used with caution. The main purpose
 * of the rotation is to be used in 90 degree increments to either flip images or turn them from
 * portrait to landscape and vice versa. In general, that part is not very well tested/used and
 * should be used with caution.
 *
 * Scaling layers can also be used to pad/unpad data or to apply an activation function explicitly,
 * just set the appropriate activation/padding and leave the scale at 1.
 */
struct ScaleLayerBuilder : ScaleLayerBuilderTempl<ScaleLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    ScaleLayerBuilder(const std::string& name) : ScaleLayerBuilderTempl<ScaleLayerBuilder>(name) {}
    using ScaleLayerBuilderTempl<ScaleLayerBuilder>::ScaleLayerBuilderTempl;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
