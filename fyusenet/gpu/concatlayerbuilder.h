//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Concatenation Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {



/**
 * @brief Templatized anchor for concatenation layer builders for GPU concatenation layers
 *
 * @see ConcatLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct ConcatLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Structure to encapsulate a single input to the concatenation
     */
    struct Input {
        Input(short chan, short pad, int fl) : channels(chan), padding(pad), flags(fl) {}
        short channels;
        short padding;
        layerflags flags;
    };

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    ConcatLayerBuilderTempl(const std::string& name):GPULayerBuilderTempl<D>(name) {
        LayerBuilderTempl<D>::type_ = LayerType::CONCAT;
    }

    /**
     * @brief Create a concatenation input
     *
     * @param channels Number of channels for the input
     * @param padding Input pading
     * @param flags Layer flags for the input layer
     *
     * @return Reference to builder
     */
    D & input(short channels, short padding, int flags = LayerFlags::NO_LAYER_FLAGS) {
        inputs_.push_back(Input(channels,padding, flags));
        LayerBuilderTempl<D>::inputChannels_ += channels;
        return *(D *)this;
    }

    std::vector<Input> inputs_;         //!< Collector for the inputs
};

/**
 * @brief Concatenation layer builder
 *
 * This class provides a builder pattern for concatenation-type layer. Unlike other layers,
 * concatenation layer have a varying amount of inputs from other layers, which can be added
 * using the input() method.
 *
 * @note The concatentation is currently restricted regarding the application of activation
 *       functions to the input. Either \e all inputs have the same activation functions or
 *       \e none of the inputs have an activation. It is currently not possible to mix
 *       these.
 */
struct ConcatLayerBuilder : ConcatLayerBuilderTempl<ConcatLayerBuilder> {
    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    ConcatLayerBuilder(const std::string& name):ConcatLayerBuilderTempl<ConcatLayerBuilder>(name) {}
};


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
