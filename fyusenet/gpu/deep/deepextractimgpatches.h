//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Image-Patch Extraction Layer (Header)
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
#include "../../gl/shaderprogram.h"
#include "deeplayerbase.h"
#include "../imgextractlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Performs the equivalent of Tensorflow's "extract image patches"
 *
 * The extract image patches operator "extrudes" a tensor by unfolding neighboring elements into
 * separate channels. For a window size of 2, the following example provides the idea:
 * @verbatim
 * 1 2 5 6  9 10    (2x6x1 input tensor with elements conveniently labeled)
 * 3 4 7 8 11 12
 *
 * [1 5 9] [2 6 10] [3 7 11] [4 8 12]
 *
 * The output has size 1x3x4
 * @endverbatim
 *
 * For multi-channel input, the channel order is arranged such that the original channels appear
 * as innermost repetition, again an example for a window size of 2:
 * @verbatim
 * [  1  2  3  4 ] [  5  6  7  8 ]     (example with two input channels IC0 and IC1)
 * [  9 10 11 12 ] [ 13 14 15 16 ]
 *       IC0             IC1
 *
 * [ 1 3 ] [ 5 7 ] [ 2 4 ] [ 6 8 ]  ...
 *   IC0     IC1     IC0     IC1
 * @endverbatim
 *
 * @warning This layer has not been used for a long time and may be subject to bugs
 */
class DeepExtractImagePatches : public DeepLayerBase {
 public:

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepExtractImagePatches(const ImgExtractLayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
    void setup() override;
    void forward(uint64_t sequenceNo, StateToken * state) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders();
    void setupNetworkPolygons(VAO *vao);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;                 //!< Shader program for the reformat operation
    unistateptr shaderState_;           //!< Associated uniform state for the #shader_
    VAO *vertexArray_ = nullptr;        //!< Pointer to vertex array object that tracks the buffer objects
    VBO *vertexBuffer_ = nullptr;       //!< Pointer to vertex buffer object for polygon vertices
    VBO *positionBuffer_ = nullptr;     //!< Pointer to vertex buffer object that halds the polygon vertex coordinates
    IBO *indexBuffer_ = nullptr;        //!< Pointer to index buffer object that defines the polygon connectivity
    int window_ = 0;                    //!< Window size (isotropic) for the reformat operation
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
