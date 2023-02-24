//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Neural Network Layer Base
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <algorithm>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "cpulayerbase.h"

namespace fyusion {
namespace fyusenet {
namespace cpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param builder GPU-specific layer builder that contains parameterization for the layer
 *
 * @param layerNumber Layer number that defines sequence position in execution
 *
 * @throws FynException in case the layer is initialized with invalid/unsupported parameters
 *
 * Parses basic information from the supplied \p builder and initializes the object accordingly.
 */
CPULayerBase::CPULayerBase(const LayerBuilder& builder, int layerNumber) : LayerBase(builder, layerNumber), CPULayerInterface() {
    device_ = compute_device::DEV_CPU;
}


/**
 * @brief Destructor
 *
 * Deallocates all (CPU) resources used (and owned) by the layer object.
 */
CPULayerBase::~CPULayerBase() {   
}


/**
 * @copydoc CPULayerInterface::addOutputBuffer
 */
void CPULayerBase::addOutputBuffer(CPUBuffer * buf, int port) {
    assert(port >= 0);
    if ((int)outputs_.size() == port) outputs_.push_back(buf);
    else if ((int)outputs_.size() > port) outputs_[port] = buf;
    else {
        outputs_.resize(port+1, nullptr);
        outputs_[port] = buf;
    }
}


/**
 * @copydoc CPULayerInterface::setInputBuffer
 */
void CPULayerBase::setInputBuffer(CPUBuffer * buf, int port) {
    assert(port >= 0);
    if ((int)inputs_.size() == port) inputs_.push_back(buf);
    else if ((int)inputs_.size() > port) inputs_[port] = buf;
    else {
        inputs_.resize(port+1, nullptr);
        inputs_[port] = buf;
    }
}


/**
 * @copydoc CPULayerInterface::clearInputBuffers
 */
void CPULayerBase::clearInputBuffers(int port) {
    if (port == -1) inputs_.clear();
    else if ((int)inputs_.size() > port) inputs_[port] = nullptr;
}


/**
 * @copydoc CPULayerInterface::clearOutputBuffers
 */
void CPULayerBase::clearOutputBuffers(int port) {
    if (port == -1) outputs_.clear();
    else if ((int)outputs_.size() > port) outputs_[port] = nullptr;
}


/**
 * @copydoc CPULayerInterface::setResidualBuffer
 */
void CPULayerBase::setResidualBuffer(CPUBuffer * buf) {
    residuals_.push_back(buf);
}

/**
 * @copydoc LayerBase::setup
 */
void CPULayerBase::setup() {
    // empty on purpose
}


/**
 * @copydoc LayerBase::cleanup
 */
void CPULayerBase::cleanup() {
    // empty on purpose
}


/**
 * @copydoc LayerBase::writeResult
 */
void CPULayerBase::writeResult(const char *fileName, bool includePadding) {
#ifdef DEBUG
    assert(includePadding == false);
    assert(outputs_.size() <= 1);           // fow now we only support one output
    if (outputs_.size() > 0) {
        outputs_.at(0)->write<float>(fileName);
    }
#endif
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
