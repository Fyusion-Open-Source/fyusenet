//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Batchnorm Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../base/bufferspec.h"
#include "../../base/batchnorminterface.h"
#include "deepfunctionlayer.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Batch-norm layer for deep format tensors
 *
 * This layer implements the batch-norm operator which basically scales and shifts the input
 * data using channel-individual scale + bias values.
 *
 * This layer should only be used in exceptional circumstances, since most layer types support
 * a fused/implicit batchnorm which is more efficient than doing it explicitly.
 *
 * @note This layer does not track any batches (our batch size is always 1 anyway), but uses
 *       fixed values obtained and stored during training (running means and variances)
 *
 * @see https://en.wikipedia.org/wiki/Batch_normalization
 */
class DeepBatchNormLayer : public DeepFunctionLayer, public BatchNormInterface {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepBatchNormLayer(const GPULayerBuilder & builder,int layerNumber);
    virtual ~DeepBatchNormLayer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    virtual void loadScaleAndBias(const float * scaleAndBias, size_t sbOffset) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupNetworkPolygons(VAO *vao) override;
    virtual void setupShaders() override;
    virtual void renderChannelBatch() override;
    virtual void beforeRender() override;
    virtual void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;                 //!< Shader program for the scaling
    unistateptr shaderState_;           //!< UniformState object for the #shader_
    float * bnScales_ = nullptr;        //!< Scaling values for batchnorm
    float * bnBias_ = nullptr;          //!< Bias values for batchnorm
    VBO * scaleAttribs_ = nullptr;      //!< VBO for batchnorm scales
    VBO * biasAttribs_ = nullptr;       //!< VBO for batchnorm biases
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
