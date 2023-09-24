//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Function Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../base/bufferspec.h"
#include "deeplayerbase.h"
#include "../gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {

namespace opengl {
  class VAO;
  class VBO;
  class IBO;
}

namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Base class for deep-tensor function layers
 *
 * This base class implements some default initializations and rendering code which is shared among
 * most of the rather simple layers that only perform unary operations (like activation or padding)
 * or binary operations (like addition) on shallow data using the GPU. Simple function-type layers
 * that use deep-tensor representation should be derived from this class.
 *
 * @see gpu::deep::DeepScaleLayer
 */
class DeepFunctionLayer : public DeepLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepFunctionLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup() override;
    void cleanup() override;
    void forward(uint64_t sequenceNo, StateToken * state) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
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
     * @brief Render an input batch of channels
     *
     * @pre The correct output framebuffer is bound to \c GL_FRAMEBUFFER target
     *
     * This function executes the rendering operation that in turn performs the actual computation
     * to be done by this layer via the employed fragment shaders.
     */
    virtual void renderChannelBatch() = 0;

    /**
     * @brief Perform misc post-rendering work
     *
     * This function is invoked by the forward() function after all rendering has been done. The
     * implementation of this function performs required cleanups / data resets in order to
     * prepare the instance for the next round of inference.
     */
    virtual void afterRender() = 0;

    virtual void setupNetworkPolygons(VAO *vao);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO *vertexArray_ = nullptr;        //!< Pointer to VAO object that maintains the %IBO and %VBO
    VBO *vertexBuffer_ = nullptr;       //!< Pointer to VBO object for proxy polygon data
    IBO *indexBuffer_ = nullptr;        //!< Pointer to IBO object for proxy polygon data
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
