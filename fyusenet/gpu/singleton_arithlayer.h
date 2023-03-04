//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Singleton Arithmetic Layer (Header)
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
#include "../gl/shaderprogram.h"
#include "gfxcontextlink.h"
#include "functionlayer.h"
#include "singleton_arithlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Layer that performs a static arithmetic operation with a singleton and a shallow tensor
 *
 * This layer performs a basic arithmetic operation on a tensor using a single value as 2nd operand.
 * The supported operations are:
 *   - adding/subtracting a value to \e all elements of a tensor
 *   - multiplying/dividing \e all elements of a tensor by a single value
 *
 * Note that in contrast to most other layers, the 2nd operand (the single value) is provided via
 * the SinglethonArithLayerBuilder in the constructor of this layer.
 */
class SingletonArithmeticLayer : public FunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    SingletonArithmeticLayer(const SingletonArithLayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void renderChannelBatch(int outPass,int numRenderTargets,int texOffset) override;
    virtual void setupShaders() override;
    virtual void beforeRender() override;
    virtual void afterRender() override;
    programptr compileShader(const char *preproc);
    unistateptr initShader(programptr shader,int renderTargets);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];          //!< Shader programs for the arithmetic operation
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];    //!< UniformState objects for the #shaders_
    ShaderProgram *currentShader_ = nullptr;            //!< Pointer to currently in-use shader
    ArithType optype_;                                  //!< Type of operation to perform
    float operand_ = 0.0f;                              //!<
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
