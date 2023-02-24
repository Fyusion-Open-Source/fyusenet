//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// DeepConvolution Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/fbo.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../base/bufferspec.h"
#include "../../base/convlayerinterface.h"
#include "../convlayerbase.h"
#include "deeptiler.h"
#include "deeplayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Base class for deep-tensor (high channel count) convolution layers
 *
 * This class implements some base functionality that is common to all/most convolution layers.
 * In particular, it contains the handling of the weight/bias data, which differs significantly
 * from the shallow tensor layers. It is not efficient to use multiple render passes with
 * changing uniforms for the deep-tensor convolution (at least not in my tests on mobile GPUs).
 * Instead, a different path is taken, which packs the convolution coefficients into textures
 * and uses a few tricks - when available.
 *
 * The texture format for the convolution coefficients is as follows:
 *   - Pixel format is \c RGBA
 *   - Texture \e height corresponds to the number of output channels multiplied by the convolution
 *     kernel size
 *   - Texture \e width corresponds to the number of input channels multiplied by the convolution
 *     kernel size (there is an additional tweak, see below for details)
 *   - Each pixel in the texture corresponds to 4 (or 8) convolution coefficients that are laid
 *     out as as part of 4x4 matrices
 *   - Four (4) consecutive pixels in a row represent a 4x4 matrix with the input channels as their
 *     column space and the output channels as their row space
 *   - Depending on the convolution kernel size, \e k neighboring 4x4 matrices horizontally
 *     represent the horizontal part of the kernel and \e k neighboring 4x4 matrices vertically
 *     represent the vertical part of the kernel
 *   - I should really put a picture here
 *
 * An additional tweak to the setup described above is the capability to contract the RAM
 * requirements by half. In order to do that, we do not use a floating-point texture, but a
 * 32-bit integer (per channel) texture. We then fit two 16-bit floating-point numbers in a
 * single channel and can reduce the texture width by 50%. This has to be decoded by the shader
 * later.
 */
// TODO (mw) it is not really good that this class is not derived from deeplayerbase, find some fix for that
class DeepConvLayerBase : public ConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepConvLayerBase(const ConvLayerBuilder& builder, int layerNumber);
    virtual ~DeepConvLayerBase();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void loadWeightsAndBiases(const float *weights, size_t offset) override;
    virtual void setup() override;
    virtual void cleanup() override;
    virtual bool isApplicable() const override;
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;

    /**
     * @brief Obtain pointer to data tiler that is used for this object
     *
     * @return Pointer to DeepTiler object that is used to compute the tiling for this layer
     */
    DeepTiler * getTiler() const {
        return tiler_;
    }

    /**
     * @brief Obtain pointer to residual tiler that is used for this object
     *
     * @return Pointer to DeepTiler object that is used to compute the tiling for the (optional)
     *         residual part of this layer
     *
     * @note The returned object may be a \c nullptr
     */
    DeepTiler * getResidualTiler() const {
        return residualTiler_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
    virtual void setupNetworkPolygons(VAO *vao);
    virtual size_t shaderPreprocessing(char *preproc,size_t maxChars);
    virtual void shaderPostprocessing(programptr shader);    
    virtual void setupFBOs() override;
    virtual void updateFBOs() override;

    /**
     * @brief Compile convolution-specific shaders
     *
     * @param preproc Existing preprocessor macros for the shader
     *
     * This function compiles and links the convolution-specific shader(s) and stores the shader
     * programs in the appropriate locations at #shader_ and #noBiasShader_ . In addition, the
     * shader state objects at #shaderState_ and #noBiasShaderState_ are initialized with the
     * implementation-specific values.
     */
    virtual void compileConvolutionShaders(const char * preproc) = 0;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    DeepTiler *tiler_ = nullptr;                //!< Pointer to texture tiler for deep tensor format (regular input / output)
    DeepTiler *residualTiler_ = nullptr;        //!< Pointer to texture tiler for deep tensor format (residual input)
    GLuint weightTexture_ = 0;                  //!< Texture handle for the convolution weights
    GLuint biasTexture_ = 0;                    //!< Texture handle for the bias data
    GLuint inputCoordTexture_ = 0;              //!< Texture handle for the input coordinates
    VAO * vertexArray_ = nullptr;               //!< Pointer to vertex array object that tracks the buffer objects
    VBO * vertexBuffer_ = nullptr;              //!< Pointer to vertex buffer object for polygon vertices / texture coordinates
    VBO * residualBuffer_ = nullptr;            //!< Pointer to vertex buffer object for polygon vertices / texture coordinates related to optional residual input
    VBO * textureOffsets_ = nullptr;            //!< Pointer to vertex buffer object that holds offsets to the weight texture to perform the convolution
    IBO * indexBuffer_ = nullptr;               //!< Pointer to index buffer object that defines the connectivity for the #vertexBuffer_
    float * postBNScales_ = nullptr;            //!< Scaling values for post-render batchnorm
    float * postBNBias_ = nullptr;              //!< Bias values for post-render batchnorm
    programptr shader_;                         //!< Convolution shader program
    programptr noBiasShader_;                   //!< Convolution hader program that does not include the network bias
    unistateptr shaderState_;                   //!< Uniform-variable state for #shader_
    unistateptr noBiasShaderState_;             //!< Uniform-variable state for #noBiasShader_

    bool mali_ = false;                         //!< Indicator flag for ARM Mali GPUs
    bool preG71_ = false;                       //!< Indicator flat for (old) ARM Mali GPUs prior to G71

    constexpr const static int DISP_TEXTURE = 4;
    constexpr const static int WEIGHT_TEXTURE = 5;
    constexpr const static int BIAS_TEXTURE = 6;

};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
