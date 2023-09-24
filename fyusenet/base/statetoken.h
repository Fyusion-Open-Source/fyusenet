//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// State Token (Header)                                                        (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <unordered_set>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerbase.h"

namespace fyusion::fyusenet {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Base class (structure) for run-specific states to be passed into the inference steps
 *
 * This class is used as base class for state-related information which is supplied to a network
 * layer in the forward() call. It may be subclassed to provide customized state information (and
 * more) to custom/specialized layers. Its main uses are:
 *   1. ability to "customize" an inference run (masking out layers for example)
 *   2. store state information between forward() calls, e.g. for sequence learning.
 *
 * @see LayerBase::forward
 */
struct StateToken {
    int seqLength = 0;                    //!< For sequence-learning layers, provides the number of tokens in the query
    int seqIndex = 0;                     //!< For sequence-learning layers, provides the current index into the sequence
    bool reset = false;                   //!< Flag indicating whether the state in stateful layers should reset prior to execution for this run
    std::unordered_set<int> maskLayers;   //!< Layer numbers to be masked out for this run
};


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:

