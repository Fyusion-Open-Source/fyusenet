//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Layer Factory
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "cpulayerfactory.h"
#include "convlayer.h"
#include "reducelayer.h"

//-------------------------------------- Global Variables ------------------------------------------


namespace fyusion {
namespace fyusenet {
namespace cpu {
//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Get name/identifier of factory backend
 *
 * @return "CPU" string
 */
std::string CPULayerFactoryBackend::getName() const {
    static std::string name("CPU");
    return name;
}


/**
 * @brief Create layer that executes on the CPU
 *
 * @param type Layer type to create
 *
 * @param builder Builder that contains the parameters for the layer
 *
 * @param layerNumber Number to assign to the layer (layers are executed in ascending number order)
 *
 * @return Pointer to the created layer
 *
 * @throws FynException in case there was a problem with the layer creation
 */
fyusenet::LayerBase * CPULayerFactoryBackend::createLayer(LayerType type,LayerBuilder * builder, int layerNumber) {
    switch (type) {
        case LayerType::CONVOLUTION2D:
            return (fyusenet::LayerBase *)createConvLayer((ConvLayerBuilder *)builder,layerNumber);
        case LayerType::REDUCE:
            return (fyusenet::LayerBase *)createReduceLayer((ReduceLayerBuilder *)builder,layerNumber);
        default:
            THROW_EXCEPTION_ARGS(FynException,"Unsupported layer type");
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 */
CPULayerFactoryBackend::CPULayerFactoryBackend():LayerFactoryBackend() {
}


/**
 * @brief Create a CPU-based convolution layer
 *
 * @param builder Builder that contains the parameters for the layr
 *
 * @param layerNumber Number to be assigned to the layer
 *
 * @return Pointer to ConvolutionLayer
 */
LayerBase * CPULayerFactoryBackend::createConvLayer(ConvLayerBuilder *builder,int layerNumber) {
    return new ConvolutionLayer(*builder,layerNumber);
}


/**
 * @brief Create L1/L2 norm/reduction layer
 *
 * @param builder Builder that contains parameters for the layer
 *
 * @param layerNumber Number to be assigned to the layer
 *
 * @return Pointer to ReduceLayer
 */
LayerBase * CPULayerFactoryBackend::createReduceLayer(ReduceLayerBuilder *builder, int layerNumber) {
    return new ReduceLayer(*builder, layerNumber);
}

} // cpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
