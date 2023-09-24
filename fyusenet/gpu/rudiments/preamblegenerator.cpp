//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Basic Shader Preprocessor Preamble Generator
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>
#include <cstdint>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#ifndef HIGH_PRECISION
#include "../../gl/glinfo.h"
#endif
#include "../../common/logging.h"
#include "preamblegenerator.h"

namespace fyusion::fyusenet::gpu::rudiments {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Idle constructor
 */
PreambleGenerator::PreambleGenerator() {
}


/**
 * @brief Constructor
 *
 * @param builder
 *
 * Reference to (GPU base version of) builder that contains the flags to be used for preamble.
 */
PreambleGenerator::PreambleGenerator(const GPULayerBuilder &builder) {
    leakyReLU_ = builder.leakyReLU_;
    lowClip_ = builder.clipLow_;
    preActMask_ = builder.preActMask_;
    highClip_ = builder.clipHigh_;
    inputPadding_ = builder.inputPadding_;
    outputPadding_ = builder.outputPadding_;
    residualPadding_ = builder.residualPadding_;
    flags_ = builder.getFlags();
}


/**
 * @brief Generate preprocessor definitions based on stored layer flags
 *
 * @param[inout] preproc User-defined preprocessor definitions
 *
 * @param maxChars Maximum number of characters available in the \p preproc string
 *
 * @return Buffer capacity left in the supplied \p preproc buffer
 *
 * This is an overloaded function provided for convenience, check
 * generatePreprocessorPreamble(layerflags, char*, size_t) for more details.
 *
 * @see generatePreprocessorPreamble(layerflags, char*, size_t)
 */
size_t PreambleGenerator::generatePreprocessorPreamble(char *preproc, size_t maxChars) const {
    return generatePreprocessorPreamble(flags_, preproc, maxChars);
}


/**
 * @brief Generate preprocessor definitions based on stored layer flags (maksed)
 *
 * @param[inout] preproc User-defined preprocessor definitions
 *
 * @param maxChars Maximum number of characters available in the \p preproc string
 *
 * @param mask Layer flags to be <b>masked out</b> from the internal mask (i.e. ignored)
 *
 * @return Buffer capacity left in the supplied \p preproc buffer
 *
 * This is an overloaded function provided for convenience, check
 * generatePreprocessorPreamble(layerflags, char*, size_t) for more details.
 *
 * @see generatePreprocessorPreamble(layerflags, char*, size_t)
 */
size_t PreambleGenerator::generatePreprocessorPreamble(char *preproc, size_t maxChars, layerflags mask) const {
    layerflags masked = flags_ & ~mask;
    return generatePreprocessorPreamble(masked, preproc, maxChars);
}



/**
 * @brief Generate preprocessor definitions based on provided layer flags
 *
 * @param flags Layer flags to be turned into preprocessor definitions
 *
 * @param[inout] preproc User-defined preprocessor definitions
 *
 * @param maxChars Maximum number of characters available in the \p preproc string
 *
 * @return Buffer capacity left in the supplied \p preproc buffer
 *
 * This function blocks the handling of preprocessor definitions in conjunction with layer-flags.
 * Based on the flags that were passed in \p flags, it appends preprocessor definitions to the
 * supplied \p preproc string. The following preprocessor strings are set for the layer flags:
 *  \li \c PRE_RELU adds the \c ACT_RELU preprocessor definition. In case a leaky ReLU is selected,
 *      and additional definition \c LEAKY_RELU with the leak value is added
 *  \li \c PRE_CLIP adds the \c ACT_CLIP preprocessor definition as well as CLIP_LOW and CLIP_HIGH
 *      with the respective values for the clipping activation function
 *  \li \c PRE_SILU adds the \c ACT_SILU preprocessor definition
 *  \li \c PRE_GELU adds the \c ACT_GELU preprocessor definition
 *  \li \c RESIDUAL_INPUT adds the \c USE_RESIDUAL preprocessor definition
 *  \li \c RELU_ON_RESIDUAL adds the \c RELU_ON_RESIDUAL preprocessor definition
 *  \li \c POST_BATCHNORM adds the \c POST_BATCHNORM preprocessor definition
 *
 *  In case no activation function was specified, \c NO_ACT is added as preprocessor definition.
 *
 *  Independent of the layer flags, a couple of additional preprocessor definitions are set, which
 *  are:
 *   - \c PIXEL_PACKING which defines the number of channels per pixel (4 usually)
 *   - \c PADDING defines the padding value on the input data
 *   - \c NO_HALF if set, indicates that 16-bit floating point data is not available as texture
 *        format
 *   - \c HIGH_PRECISION if set, indicates that high precision (full 32-bit FP) are desired and the
 *        precision qualifiers should be set to “high”
 *
 * The result of the preprocessor handling is appended to the supplied \p preproc data.
 */
size_t PreambleGenerator::generatePreprocessorPreamble(layerflags flags, char *preproc, size_t maxChars) const {
#if defined(WIN32) || defined(WIN64)
    using ssize_t = int64_t;
#endif
    char extra[80];
    assert(maxChars > 0);
    ssize_t mc = (ssize_t)activationFunctionPreamble(flags, preproc, maxChars);
    if (flags & LayerFlags::RESIDUAL_INPUT) {
        strncat(preproc, "#define USE_RESIDUAL\n", mc);
        mc = (ssize_t)(maxChars - (ssize_t)strlen(preproc));  // ouch
    }
    assert(mc > 0);
    if (flags & LayerFlags::RELU_ON_RESIDUAL) {
        strncat(preproc, "#define RELU_ON_RESIDUAL\n", mc);
        mc = (ssize_t)(maxChars - (ssize_t)strlen(preproc));  // ouch
    }
    assert(mc > 0);
    if (flags & LayerFlags::BATCHNORM_ON_RESIDUAL) {
        strncat(preproc, "#define BATCHNORM_ON_RESIDUAL\n", mc);
        mc = (ssize_t)(maxChars - (ssize_t)strlen(preproc)); // ouch
    }
    assert(mc > 0);
    if (flags & LayerFlags::POST_BATCHNORM) {
        strncat(preproc, "#define POST_BATCHNORM\n", mc);
        mc = (ssize_t)(maxChars - (ssize_t)strlen(preproc));  // ouch
    }
    snprintf(extra, sizeof(extra), "#define PIXEL_PACKING %d\n",PIXEL_PACKING);
    strncat(preproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
#ifdef HIGH_PRECISION
    strncat(preproc, "#define NO_HALF\n", mc);
    mc -= 16;
#else
    if (!GLInfo::supportsHalf()) {
        strncat(preproc, "#define NO_HALF\n", mc);
        mc -= 16;
    }
#endif
    snprintf(extra, sizeof(extra), "#define PADDING %d\n",inputPadding_);
    strncat(preproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc >= 0);
#ifdef HIGH_PRECISION
    strncat(preproc, "#define HIGH_PRECISION\n", mc);
    mc = maxChars - (ssize_t)strlen(preproc);  // ouch
#endif
    return (size_t)std::max((ssize_t)0,mc);
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Handle preprocessor flags related to activation functions
 *
 * @param flags Layer flags to be turned into preprocessor definitions
 *
 * @param[inout] preproc User-defined preprocessor definitions
 *
 * @param maxChars Maximum number of characters available in the \p preproc string
 *
 * @return Buffer capacity left in the supplied \p preproc buffer
 *
 * This function handles the activation-related preprocessor flags (e.g. ReLU, clipping etc.) and
 * appends the correct preprocessor definitions to the supplied \p preproc. The following
 * substitutions are currently done:
 *
 *  \li \c PRE_RELU adds the \c ACT_RELU preprocessor definition. In case a leaky ReLU is selected,
 *      and additional definition \c LEAKY_RELU with the leak value is added
 *  \li \c PRE_CLIP adds the \c ACT_CLIP preprocessor definition as well as CLIP_LOW and CLIP_HIGH
 *      with the respective values for the clipping activation function
 *  \li \c PRE_SILU adds the \c ACT_SILU preprocessor definition
 *  \li \c PRE_GELU adds the \c ACT_GELU preprocessor definition
 *  \li \c PRE_SIGMOID adds the \c ACT_SIGMOID preprocessor definition
 *  \li \c PRE_TANH adds the \c ACT_TANH preprocessor definition
 *
 *  In case no activation function was specified, \c NO_ACT is added as preprocessor definition.
 */
// TODO (mw) refactor this for more flexible activation
size_t PreambleGenerator::activationFunctionPreamble(layerflags flags, char *preproc, size_t maxChars) const {
#if defined(WIN32) || defined(WIN64)
        using ssize_t = int64_t;
#endif
    char extra[256];
    ssize_t mc = (ssize_t)maxChars;
    if ((flags & LayerFlags::PRE_ACT_MASK) == 0) {
        strncat(preproc, "#define NO_ACT\n", mc);
        mc -= (ssize_t)strlen(preproc);
        assert(mc > 0);
    } else {
        if (flags & LayerFlags::PRE_CLIP) {
            snprintf(extra, sizeof(extra), "#define ACT_CLIP\n#define CLIP_LOW %f\n#define CLIP_HIGH %f\n", lowClip_, highClip_);
            strncat(preproc, extra, mc);
            mc -= (ssize_t)strlen(extra);
            assert(mc > 0);
        } else if (flags & LayerFlags::PRE_GELU) {
            snprintf(extra, sizeof(extra), "#define ACT_GELU\n");
            strncat(preproc, extra, mc);
            mc -= (ssize_t)strlen(extra);
            assert(mc > 0);
        } else if (flags & LayerFlags::PRE_SILU) {
            snprintf(extra, sizeof(extra), "#define ACT_SILU\n");
            strncat(preproc, extra, mc);
            mc -= (ssize_t)strlen(extra);
            assert(mc > 0);
        } else if (flags & LayerFlags::PRE_RELU) {
            // TODO (mw) support: ACT_SIGMOID and ACT_TANH
            strncat(preproc, "#define ACT_RELU\n", mc);
            mc = (ssize_t)maxChars - (ssize_t)strlen(preproc);
            assert(mc > 0);
            if (leakyReLU_ != 0.0f) {
                snprintf(extra, sizeof(extra),"#define LEAKY_RELU %f\n", leakyReLU_);
                strncat(preproc, extra, mc);
                mc -= (ssize_t)strlen(extra);
            }
            assert(mc > 0);
        } else {
            // FIXME (mw) unsupported activation, disable it for now
            FNLOGW("Unsupported activation function, disabling activation");
            strncat(preproc, "#define NO_ACT\n", mc);
            mc = (ssize_t)maxChars - (ssize_t)strlen(preproc);
            assert(mc > 0);
        }
        snprintf(extra, sizeof(extra), "#define ACTIVATION_MASK %d\n", preActMask_);
        strncat(preproc, extra, mc);
        mc -= (ssize_t)strlen(extra);
        assert(mc > 0);
    }
    return (size_t)std::max((ssize_t)0,mc);
}


} // fyusion::fyusenet::gpu::rudiments namespace

// vim: set expandtab ts=4 sw=4:
