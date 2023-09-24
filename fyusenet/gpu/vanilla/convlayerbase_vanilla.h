//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Base Class for Generic GPU (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../gfxcontextlink.h"
#include "../../base/bufferspec.h"
#include "../convlayerbase.h"
#include "../uniformweightarray.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::vanilla {

/**
 * @brief Base class for shallow tensor convolutions on generic GPUs
 *
 * This class serves as base class for convolutions running on a rather generic GPU (well,
 * actually more like a mobile Adreno GPU and more recent Mali GPUs). All (shallow) convolution
 * shaders for generic GPUs use the same basic approach, which is to render multiple proxy polygons
 * that differ in the texture coordinates to realize the shift in the convolution. More specifically,
 * only the vertical shift is done by using multiple proxy polygons, whereas the horizontal shift is
 * implemented in the shader itself for performance reasons.
 *
 * Furthermore, the convolution shaders support multiple render targets, which are also used to
 * improve performance on the (kind of generic) target architecture. When benchmarking different
 * approaches, I noticed that MRT gives quite an advantage on the (now admittedly old) architectures
 * that I tested on.
 *
 * The convolution coefficients (weights) and biases are routed to the shader via simple uniforms,
 * not UBOs (another thing that turned out to be better in benchmarks) prior to each shader pass.
 * I suspect that the way that UBOs are implemented on the mobile archs that I tested on, are basically
 * putting them into device memory, whereas the (classical) uniforms are set as constant memory or
 * put into the register file, which decreases access time substantially.
 *
 * Last but not least, the ROP engines are used in alpha-blending mode for free accumulation of
 * the inner product that the convolution computes. In order to use the blending trick, non-linear
 * activation functions at the end of the computation are impossible and I opted to simply
 * move those to the next layer in the chain (in the data fetch stage), which works quite fine.
 *
 * @see vanilla::ConvLayer1x1, vanilla::ConvLayer3x3, vanilla::ConvLayer5x5, vanilla::ConvLayer7x7
 * @see vanilla::ConvLayer9x9
 */
class ConvLayerBase : public gpu::ConvLayerBase {
 public:
    constexpr static int VEC_OVERHEAD = 3;
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit ConvLayerBase(const ConvLayerBuilder & builder);
    ConvLayerBase(const ConvLayerBuilder & builder, int layerNumber);
    ConvLayerBase(const GPULayerBuilder & builder, int layerNumber);
    ~ConvLayerBase() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
    void setup() override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    void loadParameters(const ParameterProvider *weights) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------

    virtual size_t shaderPreprocessing(char *preproc, size_t maxChars);
    virtual void setupNetworkPolygons(VAO *vao, int kernel);
    void setupFBOs() override;
    void updateFBOs() override;
    virtual void setBias(int outPass, const UniformWeightArray *bias);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    UniformWeightArray *weights_ = nullptr;         //!< Weight/Bias/BN data that is required to operate this layer (see #loadParameters)
    VAO * vertexArray_ = nullptr;                   //!< Pointer to vertex-array object which maintains the VBO / IBO config
    VBO * vertexBuffer_ = nullptr;                  //!< Pointer to VBO for the polygons used in convolution
    VBO * residualBuffer_ = nullptr;                //!< Pointer to VBO for the polygons used for the residual
    IBO * indexBuffer_ = nullptr;                   //!< Pointer to IBO used for convolution (and residual) polygons
    float *zeroBias_ = nullptr;                     //!< As the name implies, bias vector with all zeros
    int maxRenderTargets_ = 0;                      //!< Maximum number of render targets that can be used by this layer
    float sourceStep_ = 1.0f;                       //!< Defines the step-width of the convolution (source-side) for fractional convolutions
    bool mali_ = false;                             //!< Flag that is set when an ARM Mali GPU was detected
    bool preG71_ = false;                           //!< Flag that is set when an ARM Mali GPU prior to the G-71 model (e.g. T-880) was detected
};

} // fyusion::fyusenet::gpu::vanilla namespace

// vim: set expandtab ts=4 sw=4:
