//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Transpose-Convolutional Layer w/ 2x2 mask (Header)
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
namespace fyusion::fyusenet::gpu::deep {

/**
 * @brief Transpose convolution layer for a 2x2 convolution kernel on deep tensor data
 *
 * This is an implementation of a 2x2 transpose convolution layer, which is usually used for
 * upsampling purposes.
 *
 * @warning The current implementation is fixed to stride 2 (i.e. performing upsampling).
 *
 * @see DeepTransConvLayerBase
 */

class DeepTransConvLayer2x2 : public DeepTransConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepTransConvLayer2x2(const ConvLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void compileConvolutionShaders(const char *preproc) override;
    void renderPass(int pass) override;
    unistateptr initShader(programptr shader);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;                         //!< Convolution shader program
    programptr noBiasShader_;                   //!< Convolution shader program that does not include the network bias
    unistateptr shaderState_;                   //!< Uniform-variable state for #shader_
    unistateptr noBiasShaderState_;             //!< Uniform-variable state for #noBiasShader_
};

} // fyusion::fyusenet::gpu::deep namespace


// vim: set expandtab ts=4 sw=4:
