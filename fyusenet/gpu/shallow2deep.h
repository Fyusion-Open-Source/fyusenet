//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Shallow to Deep Layer Converter (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/uniformstate.h"
#include "../gl/fbo.h"
#include "../gl/vbo.h"
#include "../gl/ibo.h"
#include "../gl/vao.h"
#include "gfxcontextlink.h"
#include "../base/bufferspec.h"
#include "deep/deeplayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu {

/**
 * @brief Convert shallow tensor format to deep tensor format
 *
 * FyuseNet differentiates between tensors with a low channel count, which we call \e shallow tensors,
 * and tensors with a high channel count, which we call \e deep tensors. The specifics for those
 * are laid out @ref GPULayerBase "here". The purpose of this class it so convert the tensor data from
 * the \e shallow representation to the \e deep representation format.
 */
class Shallow2DeepLayer : public deep::DeepLayerBase {
 public:

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    Shallow2DeepLayer(const GPULayerBuilder & builder, int layerNumber);
    ~Shallow2DeepLayer() override = default;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void setup() override;
    void cleanup() override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupNetworkPolygons(VAO *vao);
    void setupShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO * vertexArray_ = nullptr;    //!< Pointer to vertex array object that tracks the buffer objects
    VBO * vertexBuffer_ = nullptr;   //!< Pointer to vertex buffer object for polygon vertices / texture coordinates
    IBO * indexBuffer_ = nullptr;    //!< Pointer to index buffer object that defines the polygon connectivity
    VBO * texUnitBuffer_ = nullptr;  //!< Pointer to buffer that controls which texture unit is used for texture transfer on the input side
    programptr shader_;              //!< Shader program for the conversion
    unistateptr shaderState_;        //!< State to be attached to the #shader_
    int maxInputTextures_ = 8;       //!< Max number of input textures which can be used, set to 8 initially and will be the minimum of 8 and what the system supports
    int numRenderPasses_ = 1;        //!< Number of render passes required for execution
};


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
