//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Batch Normlization Layer Interface (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cstdlib>

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {

/**
 * @brief Basic interface for stand-alone batchnorm layers
 *
 * This interface defines the batchnorm data interface for all sub-types of batchnorm layers.
 */
class BatchNormInterface {
 public:
    /**
     * @brief Load scale and bias values for a batchnorm operator
     *
     * @param scaleBias Pointer to 32-bit floating-point data that contains the scale and bias for
     *                  each channel (see long description for format)
     *
     * @param sbOffset Optional offset to the supplied \p scaleBias pointer to start reading from
     *
     * This function reads the scale and bias data from the supplied pointer. It assumes that the
     * data is formatted such that all scale values for each output channel come first, followed by
     * the bias data for each output channel as 2nd block.
     */
    virtual void loadScaleAndBias(const float *scaleAndBias, size_t sbOffset=0) = 0;
};

}  // fyusenet namespace
}  // fyusion namespace

// vim: set expandtab ts=4 sw=4:

