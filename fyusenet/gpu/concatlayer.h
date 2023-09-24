//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Concatenation Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/uniformstate.h"
#include "../gl/fbo.h"
#include "../gl/vao.h"
#include "../gl/vbo.h"
#include "../gl/ibo.h"
#include "../base/bufferspec.h"
#include "concatlayerbuilder.h"
#include "gpulayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Perform concatenation of several shallow format tensors into a target (shallow) tensor
 *
 * This class implements a concatenation layer, which is able to concatenate an variable amount
 * of input tensors into a single output tensor. Due to the way shallow tensors are stored, the
 * concatenation can either be a simple no-op, or involves one or multiple render steps.
 *
 * For example, concatenating 3 tensors with 8, 12 and 24 channels can simply be done by just
 * using the textures as input for the next layer and is therefore a no-op. In general, all
 * tensors with channels being a multiple of 4 can essentially be concatenated for free.
 *
 * The story is slighly different for tensors with channel sizes not a multiple of 4. For these
 * tensors, a consolidation render into the target texture(s) has to be done, packing the
 * concatenees together.
 *
 * @note The concatentation is currently restricted regarding the application of activation
 *       functions to the input. Either \e all inputs have the same activation functions or
 *       \e none of the inputs have an activation. It is currently not possible to mix
 *       these.
 */
class ConcatLayer : public GPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ConcatLayer(const ConcatLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup() override;
    void cleanup() override;
    virtual void addInput(int inputChannels,int inputPadding);
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    void forward(uint64_t sequenceNo, StateToken * state) override;
    [[nodiscard]] int numInputPorts() const override;
    [[nodiscard]] int getPortChannelIndex(int port) const override;
    [[nodiscard]] int numInputChannels(int port=0) const override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void addInputTexture(GLuint textureID, int channelIndex) override;
    void setupFBOs() override;
    void updateFBOs() override;
    virtual void setupShaders();
    virtual void setupNetworkPolygons(VAO *vao);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO * vertexArray_ = nullptr;              //!< Pointer to vertex-array object which maintains the VBO / IBO config
    VBO * vertexBuffer_ = nullptr;             //!< Pointer to VBO for the proxy polygons
    IBO * indexBuffer_ = nullptr;              //!< Pointer to IBO used for proxy polygons
    std::vector<int> portChannels_;            //!< List of channels per port for the inputs, a \e port refers to one input tensor
    std::vector<int> portOffsets_;             //!< Offsets into the output tensor per port
    bool consolidationRender_ = false;         //!< Indicator if a consolidation render is required, or we can just simply stack the textures
    programptr concatShaders_[3*4];            //!< Set of shaders required to do consolidation rendering
    programptr defaultShader_ = nullptr;       //!< Default shader to use for consolidation rendering
    unistateptr defaultShaderState_;           //!< Uniform state for #defaultShader_
    unistateptr concatShaderStates_[3*4];      //!< Uniform states for #concatShaders_
    int currentInputChannels_ = 0;             //!<
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
