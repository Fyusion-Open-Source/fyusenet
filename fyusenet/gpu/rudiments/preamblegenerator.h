//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Basic Shader Preprocessor Preamble Generator (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../base/layerbuilder.h"
#include "../gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::rudiments {

/**
 * @brief 
 *
 *
 */
// TODO (mw) docs
class PreambleGenerator {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    PreambleGenerator();
    explicit PreambleGenerator(const GPULayerBuilder& builder);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    size_t generatePreprocessorPreamble(char *preproc, size_t maxChars) const;
    size_t generatePreprocessorPreamble(char *preproc, size_t maxChars, layerflags mask) const;
    size_t generatePreprocessorPreamble(layerflags flags, char *preproc, size_t maxChars) const;
    size_t activationFunctionPreamble(layerflags flags, char *preproc, size_t maxChars) const;

   // ------------------------------------------------------------------------
    // Public member variables
    // ------------------------------------------------------------------------
    int inputPadding_ = 0;                           //!< Defines border around spatial dimensions of input tensor
    int outputPadding_ = 0;                          //!< Defines border around spatial dimensions of output tensor
    int residualPadding_ = 0;                        //!< Defines border around spatial dimensions of residual-input tensor
    float leakyReLU_ = 0.0f;                         //!< Optional leak value for leaky ReLUs (fused on GPUs)
    float lowClip_ = 0.0f;                           //!< For clipping-type activation function (lower end of clip)
    float highClip_ = 0.0f;                          //!< For clipping-type activation function (upper end of clip)
    uint16_t preActMask_ = 0xFFFF;                   //!< Optional activation mask which switches the prefix activation on the inputs
    layerflags flags_ = LayerFlags::NO_LAYER_FLAGS;  //!< Misc flags for this layer
};


} // fyusion::fyusenet::gpu::rudiments namespace

// vim: set expandtab ts=4 sw=4:
