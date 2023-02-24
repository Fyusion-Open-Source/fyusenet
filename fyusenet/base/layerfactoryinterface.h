//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Layer Factory Interface  (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------


namespace fyusion {
namespace fyusenet {
//------------------------------------- Public Declarations ----------------------------------------

struct LayerBuilder;

/**
 * @brief Interface for the layer factory to be used by the builders
 */
class LayerFactoryInterface {
 public:
    virtual void pushBuilder(LayerBuilder *) = 0;
};


} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:

