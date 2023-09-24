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


namespace fyusion::fyusenet {

//------------------------------------- Public Declarations ----------------------------------------

struct LayerBuilder;

namespace builder_internal {
   class Pusher;
}


/**
 * @brief Interface for the layer factory to be used by the builders
 */
class LayerFactoryInterface {
    friend class builder_internal::Pusher;
 protected:
    virtual void pushBuilder(LayerBuilder *) = 0;
};


namespace builder_internal {
    class Pusher {
     public:
        static void push(LayerFactoryInterface * factory, LayerBuilder * builder) {
            factory->pushBuilder(builder);
        }
    };
}


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:

