//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Transpose-Convolutional Layer w/ 3x3 mask (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <mutex>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "deeptransconvlayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Transpose convolution layer for a 3x3 convolution kernel on deep tensor data
 *
 * This is an implementation of a 3x3 transpose convolution layer, which is usually used for
 * upsampling purposes.
 *
 * @warning The current implementation is fixed to stride 2 (i.e. performing upsampling).
 *
 * @see DeepTransConvLayerBase
 */

class DeepTransConvLayer3x3 : public DeepTransConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepTransConvLayer3x3(const ConvLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void compileConvolutionShaders(const char *preproc) override;
    unistateptr initShader(programptr shader);
    virtual void renderPass(int pass) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;                         //!< Convolution shader program
    programptr noBiasShader_;                   //!< Convolution shader program that does not include the network bias
    unistateptr shaderState_;                   //!< Uniform-variable state for #shader_
    unistateptr noBiasShaderState_;             //!< Uniform-variable state for #noBiasShader_
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
