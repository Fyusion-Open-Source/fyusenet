//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Function Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/fbo.h"
#include "../gl/vbo.h"
#include "../gl/ibo.h"
#include "../gl/vao.h"
#include "../base/bufferspec.h"
#include "gpulayerbase.h"
#include "../base/layerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu {

/**
 * @brief Base class for simple function-type layers that perform unary/binary operations
 *
 * This base class implements some default initializations and rendering code which is shared among
 * most of the rather simple layers that only perform unary operations (like activation or padding)
 * or binary operations (like addition) on shallow data using the GPU. Simple function-type layers
 * should be derived from this class.
 *
 * @see gpu::SigmoidLayer, gpu::CastLayer, gpu::SingletonArithmeticLayer, gpu::BatchNormLayer
 * @see gpu::vanilla::ScaleLayer
 */
class FunctionLayer : public GPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    FunctionLayer(const GPULayerBuilder& builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup() override;
    void cleanup() override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    void forward(uint64_t sequence, StateToken * state) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    /**
     * @brief Compile shaders that implement the actual layer functionality
     *
     * This function obtains required shaders from the resource system, compiles/caches these shaders
     * and performs base initializations on them.
     */
    virtual void setupShaders() = 0;

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

    void setupFBOs() override;
    void updateFBOs() override;
    void setupVBO(VAO *vao);
    void setupIBO(VAO *vao);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO *vertexArray_ = nullptr;        //!< Pointer to vertex array object that tracks the buffer objects
    VBO *vertexBuffer_ = nullptr;       //!< Pointer to vertex buffer object for polygon vertices / texture coordinates
    IBO *indexBuffer_ = nullptr;        //!< Pointer to index buffer object that defines the connectivity for the #vertexBuffer_
    int maxRenderTargets_;              //!< Maximum number of render targets to use for this system
    bool isSequence_ = false;           //!< Set to true if this (and derived) layers are to be used on sequence-formatted textures
};

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
