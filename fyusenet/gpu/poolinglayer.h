//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Pooling Layer Base Class (Header)
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
#include "gpulayerbase.h"
#include "poollayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Base class for shallow tensor-data pooling layers
 *
 * This class provides an interface for all kinds of pooling layers. It contains a few default
 * initializations and a simple render loop which makes use of an internal interface to which the
 * actual pooling layers have to be taylored.
 *
 * @see MaxPoolLayer, AvgPoolLayer
 */
class PoolingLayer : public GPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    PoolingLayer(const PoolLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void setup() override;
    virtual void cleanup() override;
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    virtual void forward(uint64_t sequence) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
     /**
     * @brief Perform misc pre-rendering initializations
     *
     * This function is invoked by the forward() function prior to performing any rendering and
     * after prepareRender() has been called, which sets the ROP to the correct mode and also
     * adjusts the viewport. The implementation of this function performs inits and adjustments
     * that are specific to the particular layer instance (for example activating shaders and
     * adjusting uniforms).
     */
    virtual void beforeRender() = 0;

     /**
     * @brief Perform misc post-rendering work
     *
     * This function is invoked by the forward() function after all rendering has been done. The
     * implementation of this function performs required cleanups / data resets in order to
     * prepare the instance for the next round of inference.
     */
    virtual void afterRender() = 0;

    /**
     * @brief Create shader state for supplied shader
     *
     * @param shader Shader to create a uniform state object for
     * @param renderTargets Number of render targets for the \p shader
     *
     * @return Shared pointer to UniformState object that maps values to the uniforms of a shader
     */
    virtual unistateptr initShader(programptr shader,int renderTargets)=0;


    /**
     * @brief Render an input batch of channels
     *
     * @param outPass Output pass number, starts at 0 for the first pass and is increased with
     *                every set of render targets, until all output channels have been covered
     *
     * @param numRenderTargets Number of simultaneous render targets for this pass. Will be 1 for
     *                         the default single render target and up to #maxRenderTargets_ in
     *                         case multiple render targets are supported by the system
     *
     * @param texOffset Offset in the texture list for the \e input textures
     *
     * @pre The correct output framebuffer is bound to \c GL_FRAMEBUFFER target
     *
     * This function executes the rendering operation that in turn performs the actual computation
     * to be done by this layer via the employed fragment shaders. As this function may be called
     * multiple times for targets that have a medium to large amount of channels, it keeps track
     * of the input/output channels via the \p outPass and \p texOffset parameters. These are
     * increased for every rendering pass. In addition, the \p numRenderTargets parameter controls
     * how many multiple render targets should be rendered into during that pass.
     */
    virtual void renderChannelBatch(int outPass, int numRenderTargets, int texOffset)=0;


    /**
     * @brief Compile pooling-specific shaders
     *
     * @param preproc Existing preprocessor macros for the shader
     *
     * @return Shared pointer to ShaderProgram instance
     *
     * This function compiles and links the pooling-specific shader(s) and stores the shader
     * programs in the appropriate variables in #shaders_. In addition, the shader state
     * objects at #shaderStates_ initialized with the implementation-specific values.
     */
    virtual programptr compileShader(const char *preproc) = 0;

    virtual void setupShaders();
    void setupVBO(VAO *vao);
    void setupIBO(VAO *vao);
    virtual void setupFBOs() override;
    virtual void updateFBOs() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int poolSize_[2] = {1, 1};                          //!< Pool size (x, y direction)
    int downsample_[2] = {1, 1};                        //!< Downsampling stride
    ShaderProgram *currentShader_ = nullptr;            //!< Pointer to currently active shader
    VAO *vertexArray_ = nullptr;                        //!< Pointer to vertex-array object which maintains the VBO / IBO config
    VBO *vertexBuffer_ = nullptr;                       //!< Pointer to VBO for the polygons used in the layer
    IBO *indexBuffer_ = nullptr;                        //!< Pointer to IBO used for the polygons
    int maxRenderTargets_ = 1;                          //!< Maximim number of render targets that can be used by this layer
    programptr shaders_[FBO::MAX_DRAWBUFFERS];          //!< Shared pointers to shader programs used for rendering
    unistateptr shaderStates_[FBO::MAX_DRAWBUFFERS];    //!< States that are attached to the #shaders_
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
