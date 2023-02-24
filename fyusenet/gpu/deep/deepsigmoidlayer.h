//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Sigmoid Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------
#ifndef _I_FYN_DEEPSIGMOIDLAYER_H
#define _I_FYN_DEEPSIGMOIDLAYER_H
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

#ifdef ANDROID
#include <GLES3/gl3.h>
#else
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <OpenGL/glext.h>
#else
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#endif

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../base/bufferspec.h"
#include "deepfunctionlayer.h"
#include "../scalelayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Layer that maps input data with a sigmoid function for deep tensors
 *
 * This layer maps all input data element-wise using a sigmoid function, using the following
 * mapping:
 *
 *  \f[ f(x) = \frac{1}{1+e^{-x}} \f]
 *
 * Other than padding, the result is not reformatted in any way.
 */
class DeepSigmoidLayer : public DeepFunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepSigmoidLayer(const GPULayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;    
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupShaders() override;
    virtual void renderChannelBatch() override;
    virtual void beforeRender() override;
    virtual void afterRender() override;
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the pooling
    unistateptr shaderState_;     //!< UniformState object for the #shader_
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace
#endif

// vim: set expandtab ts=4 sw=4:
