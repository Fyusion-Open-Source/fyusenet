//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform (GL) Weight Array
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------


#include "uniformweightarray.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * Pretty much idle.
 */
UniformWeightArray::UniformWeightArray() {
}

/**
 * @brief Destructor
 *
 * Releases all (CPU) memory held by weights/biases/batchnorm data.
 */
UniformWeightArray::~UniformWeightArray() {
    delete [] biasData_;
    delete [] weightData_;
    delete [] bnBias_;
    delete [] bnScale_;
    biasData_ = nullptr;
    weightData_ = nullptr;
    bnBias_ = nullptr;
    bnScale_ = nullptr;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
