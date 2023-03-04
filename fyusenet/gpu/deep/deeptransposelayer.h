//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Image Transposition Layer Layer (Header)
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
#include "deepfunctionlayer.h"
#include "../transposelayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Image transposition layer for deep channel configurations
 *
 */
class DeepTransposeLayer : public DeepLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepTransposeLayer(const TransposeLayerBuilder & builder,int layerNumber);
    ~DeepTransposeLayer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void forward(uint64_t sequence) override;
    virtual void setup() override;
    virtual void cleanup() override;
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;

 protected:

    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupNetworkPolygons(VAO *vao);
    void setupShaders();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;
    unistateptr shaderState_;
    DeepTiler * outTiler_ = nullptr;
    VAO *vertexArray_ = nullptr;
    VBO *vertexBuffer_ = nullptr;
    IBO *indexBuffer_ = nullptr;
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
