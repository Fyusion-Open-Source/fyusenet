//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Transpose-Convolution Layer Base Class (Header)
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
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../base/bufferspec.h"
#include "../convlayerbase.h"
#include "deeptiler.h"
#include "deepconvlayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Base class for deep transpose convolution layers
 *
 * This class serves as base/interface for transposed convolution layers on deep tensor data. In
 * contrast to standard convolutional layers, the transposed convolution is often used for upsampling
 * purposes (sometimes called deconvolution) performs a "broadcasting" operation on the input tensor,
 * akin to a Kronecker product, by multiplying the kernel with each element in the input tensor and
 * adding it to the output tensor. When performing upsampling, the upsampling stride determines the
 * spacing between the multiplied kernel elements in the output tensor.
 *
 * An implementation of a transpose convolution in a fragment shader is a tiny bit tricky due to the
 * broadcasting nature of the operator. The implementations derived from this class make use of a
 * stencil buffer for the broadcasting operation.
 *
 */
class DeepTransConvLayerBase : public DeepConvLayerBase {
 public:
    enum {
        DISP_TEXTURE = 4,       //!< Texture unit for displacement texture
        WEIGHT_TEXTURE = 5,     //!< Texture unit for weight texture
        BIAS_TEXTURE = 6        //!< Texture unit for bias texture
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepTransConvLayerBase(const ConvLayerBuilder & builder, int layerNumber);

    virtual void setup() override;
    virtual void cleanup() override;
    virtual void forward(uint64_t sequence) override;
    virtual void loadWeightsAndBiases(const float *biasAndWeights, size_t offset) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual size_t shaderPreprocessing(char *preproc,size_t maxChars) override;
    virtual void setupNetworkPolygons(VAO *vao) override;
    void setupStencilBuffer();

    /**
     * @brief Execute a single render pass (4 passes in total)
     *
     * @param pass Pass number, starts at 0
     *
     * The individual implementations of this function perform the actual rendering of each single
     * pass (4 of them in total) in order to fill the target tensor with the correct data. Specific
     * shaders are bound for each pass.
     */
    virtual void renderPass(int pass) = 0;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GLuint stencilBuffer_ = 0;                  //!< GL handle for renderbuffer that contains the stencil
    int upsample_[2] = {2, 2};                  //!< Upsampling parameters (currently we only support 2 here)

    constexpr const static int PASS = 1;        //!< Uniform location enumerator for the "pass" uniform in all derived shaders

};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
