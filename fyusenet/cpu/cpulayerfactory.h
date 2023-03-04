//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Layer Factory (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../base/layerfactory.h"
#include "convlayerbuilder.h"
#include "reducelayerbuilder.h"

namespace fyusion {
namespace fyusenet {
namespace cpu {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Producer backend for CPU-based network layers
 *
 * This class serves as backend for layers that execute on the CPU. As FyuseNet is not meant to be
 * used on the CPU excessively, the support for layer types here is very narrow.
 *
 * In case future versions of FyuseNet shall make use of a decent CPU fallback, more layers need
 * to be added here.
 */
class CPULayerFactoryBackend : public LayerFactoryBackend {
    friend class LayerFactory;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    CPULayerFactoryBackend();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual std::string getName() const override;
    virtual fyusenet::LayerBase * createLayer(LayerType type, LayerBuilder * builder, int layerNumber) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    LayerBase * createConvLayer(ConvLayerBuilder *builder,int layerNumber);
    LayerBase * createReduceLayer(ReduceLayerBuilder * builder, int layerNumber);
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
