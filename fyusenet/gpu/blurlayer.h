//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Blurring Layer (Header)
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
#include "gfxcontextlink.h"
#include "../base/bufferspec.h"
#include "functionlayer.h"
#include "../base/layerfactory.h"
#include "blurlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu {

/**
 * @brief Simple spatial blur layer (Gaussian/Box) for shallow tensors
 *
 * This class implements a spatial blur layer which applies either a Gaussian blur or a box-filter
 * with \b odd kernel sizes (even kernel sizes are not supported and lead to an exception).
 *
 * @note The shader implementation for this layer is a quite straightforward one and is not optimized
 *       for larger blur kernels. We propose to restrict the blur kernel size to 5, definitely not
 *       exceed 7. If larger kernels are desired, a smarter implementation will be necessary.
 *
 * @warning Make sure to use appropriate padding depending on the supplied kernel size
 */
class BlurLayer : public FunctionLayer {
 public:
    constexpr static int MAX_KERNEL_SIZE = 21;
    constexpr static int SHADER_WEIGHTS = 1;
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    BlurLayer(const BlurLayerBuilder & builder,int layerNumber);
    ~BlurLayer() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void setup() override;
    virtual void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void computeAverageWeights();
    void computeGaussianWeights();
    virtual void setupShaders() override;
    programptr compileShader(const char *preproc);
    virtual void beforeRender() override;
    virtual void afterRender() override;
    virtual void renderChannelBatch(int outPass, int numRenderTargets, int texOffset) override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shaders_[FBO::MAX_DRAWBUFFERS];         //!< Shader instance (shared) pointers (different shaders for different number of render targets)
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];   //!< Shader states that memorize the shader states of the #shaders_
    ShaderProgram *currentShader_ = nullptr;           //!< Raw pointer to currently active/in-use shader
    int kernelSize_;                                   //!< Blur kernel size
    BlurKernelType blurType_;                          //!< Blur kernel type (Gaussian or box-filter)
    float *kernelWeights_ = nullptr;                   //!< Pointer to computed kernel weights to be used in the shader
};

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
