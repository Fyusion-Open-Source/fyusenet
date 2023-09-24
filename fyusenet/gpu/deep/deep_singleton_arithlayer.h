//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Singleton Arithmetic Layer (Header)
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
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../base/bufferspec.h"
#include "deepfunctionlayer.h"
#include "../singleton_arithlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::deep {


/**
 * @brief Layer that performs a static arithmetic operation with a singleton and a deep tensor
 *
 * This layer performs a basic arithmetic operation on a tensor using a single value as 2nd operand.
 * The supported operations are:
 *   - adding/subtracting a value to \e all elements of a tensor
 *   - multiplying/dividing \e all elements of a tensor by a single value
 *
 * Note that in contrast to most other layers, the 2nd operand (the single value) is provided via
 * the SinglethonArithLayerBuilder in the constructor of this layer.
 */
class DeepSingletonArithmeticLayer : public DeepFunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepSingletonArithmeticLayer(const SingletonArithLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;
    void beforeRender() override;
    void renderChannelBatch() override;
    void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the arithmetic operation
    unistateptr shaderState_;     //!< UniformState object for the #shader_
    ArithType optype_;            //!< Type of operation to perform
    float operand_ = 0.0f;        //!< The singleton operand to use with the tensor
};

} // fyusion::fyusenet::gpu::deep namespace


// vim: set expandtab ts=4 sw=4:
