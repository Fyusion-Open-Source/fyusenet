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
#include "../convlayerbase.h"
#include "deeptiler.h"
#include "deeplayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::deep {


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
 *     out as part of 4x4 matrices
 *   - Four (4) consecutive pixels in a row represent a 4x4 matrix with the input channels as their
 *     column space and the output channels as their row space
 *   - Depending on the convolution kernel size, \e k neighboring 4x4 matrices horizontally
 *     represent the horizontal part of the kernel and \e k neighboring 4x4 matrices vertically
 *     represent the vertical part of the kernel
 *   - I should really put a picture here
 *
 * An additional tweak to the setup described above is the capability to contract the VRAM
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
    DeepConvLayerBase(const GPULayerBuilder& builder, int layerNumber);
    ~DeepConvLayerBase() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void loadParameters(const ParameterProvider *weights) override;
    void setup() override;
    void cleanup() override;
    [[nodiscard]] bool isApplicable() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;

    /**
     * @brief Obtain pointer to data tiler that is used for this object
     *
     * @return Pointer to DeepTiler object that is used to compute the tiling for this layer
     */
    [[nodiscard]] DeepTiler * getTiler() const {
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
    [[nodiscard]] DeepTiler * getResidualTiler() const {
        return residualTiler_;
    }

    void writeResult(const char *fileName, bool includePadding) override;
    void copyResult(float *memory, bool includePadding) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;
    virtual void setupNetworkPolygons(VAO *vao);
    virtual size_t shaderPreprocessing(char *preproc,size_t maxChars);
    virtual void shaderPostprocessing(programptr & shader);
    void setupFBOs() override;
    void updateFBOs() override;
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;

    /**
     * @brief Compile convolution-specific shaders
     *
     * @param preproc Existing preprocessor macros for the shader
     *
     * This function compiles and links the convolution-specific shader(s) and stores the shader
     * programs and the states (if any) in the appropriate locations.
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

    bool mali_ = false;                         //!< Indicator flag for ARM Mali GPUs
    bool preG71_ = false;                       //!< Indicator flag for (old) ARM Mali GPUs prior to G71
    bool largeDilation_ = false;                //!< Indicator if dilation is outside of the GLSL textureOffset operation
    bool halfSupport_ = false;                  //!< Indicator if 16-bit FP is supported on the platform

    constexpr const static int DISP_TEXTURE = 4;
    constexpr const static int WEIGHT_TEXTURE = 5;
    constexpr const static int BIAS_TEXTURE = 6;

};

} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
