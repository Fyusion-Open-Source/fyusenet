//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Transpose Convolution Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/fbo.h"
#include "../../gl/uniformstate.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../base/bufferspec.h"
#include "../../gpu/uniformweightarray.h"
#include "../../base/bufferspec.h"
#include "convlayerbase_vanilla.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::vanilla {

/**
 * @brief Base class for transpose convolution layers
 *
 * This class serves as base/interface for transposed convolution layers. In contrast to standard
 * convolutional layers, the transposed convolution is often used for upsampling purposes (sometimes
 * called deconvolution) performs a “broadcasting” operation on the input tensor, akin to a Kronecker
 * product, by multiplying the kernel with each element in the input tensor and adding it to the
 * output tensor. When performing upsampling, the upsampling stride determines the spacing between
 * the multiplied kernel elements in the output tensor.
 *
 * An implementation of a transpose convolution in a fragment shader is a tiny bit tricky due to the
 * broadcasting nature of the operator. The implementations derived from this class make use of a
 * stencil buffer for the broadcasting operation. Currently, the transpose convolution layers in
 * FyuseNet only support stride-2 transpose-convolutions, which performs a "convoluted upsampling"
 * of the input tensor by a factor of 2 along both spatial dimensions. The fixed 2-fold upsampling
 * basically leads to 4 different configurations which are encoded in a stencil-buffer and 4 specialized
 * shaders for each of the configurations. These configurations are referred to as stratum / strata
 * internally.
 *
 * @see vanilla::TransConvLayer2x2, vanilla::TransConvLayer3x3, ConvTransWeightArray2x2xNxM
 * @see TransConvWeightArray3x3xNxM
 */
class TransConvLayerBase : public ConvLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    TransConvLayerBase(const ConvLayerBuilder& builder, int layerNumber);
    ~TransConvLayerBase() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void setup() override;
    void cleanup() override;

 protected:

    /** Symbols for uniform locations in the shader(s) */
    enum {
        COEFFICIENTS = 1,
        BIAS,
        BATCHNORM_DATA
    };

    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] unistateptr configureShader(const programptr& shader, int stratum) const;
    void performInputPasses(UniformWeightArray *weights, int outputPass);
    void setupNetworkPolygons(VAO *vao, int kernel) override;
    void setupStencilBuffer();
    void setBias(int outPass,const UniformWeightArray *bias) override;
    size_t shaderPreprocessing(char *preproc, size_t maxChars) override;
    void setupFBOs() override;
    void updateFBOs() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int maxRenderTargets_ = 1;                    //!< For multiple render targets, determine the maximum number of MRTs that can be used on the platform
    int upsample_ = 2;                            //!< Upsampling factor (isotropic) for the convolution
    UniformWeightArray * weights_ = nullptr;      //!< Pointer to convolution weight data
    GLuint stencilBuffer_ = 0;                    //!< Stencil for stratified deconvolution
    VBO * coordBuffer_ = nullptr;                 //!< VBO for polygon coordinates
    VBO * textureBuffer_ = nullptr;               //!< VBO for texture coordinates
    IBO * indexBuffer_ = nullptr;                 //!< IBO for indexing coordinates
    VAO * vertexArray_ = nullptr;                 //!< VAO for vertex- and index-buffers
    std::vector<programptr> shaders_[4];          //!< Set of shader programs
    std::vector<unistateptr> shaderStates_[4];    //!< Uniform states of the #shaders_

    constexpr const static int VEC_OVERHEAD = 3;
    constexpr const static int NUM_STRATA = 4;

};

} // fyusion::fyusenet::gpu::vanillavanilla namespace

// vim: set expandtab ts=4 sw=4:
