//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Pooling Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

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
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#endif

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../base/bufferspec.h"
#include "deeptiler.h"
#include "deeplayerbase.h"
#include "../poollayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Base class for pooling layers operating on the deep-channel tensor format
 *
 * This class provides an interface for all kinds of pooling layers. It contains a few default
 * initializations and a simple render loop which makes use of an internal interface to which the
 * actual pooling layers have to be taylored.
 *
 * @see DeepMaxPoolLayer, DeepAvgPoolLayer, DeepGlobalPoolLayer
 */
class DeepPoolingLayer : public DeepLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepPoolingLayer(const PoolLayerBuilder & builder,int layerNumber);

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
     * @brief Compile shaders that implement the actual layer functionality
     *
     * This function obtains required shaders from the resource system, compiles/caches these shaders
     * and performs base initializations on them.
     */
    virtual void setupShaders() = 0;

    /**
     * @brief Render an input batch of channels
     *
     * This function executes the rendering operation that in turn performs the actual computation
     * to be done by this layer via the employed fragment shaders.
     */
    virtual void renderChannelBatch() = 0;

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
     * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
     *
     * @param vao Pointer to vertex array object that the resulting VBO and IBO are tied to
     *
     * @pre The supplied \p vao vertex array object to be used with this VBO is already bound
     *
     * As fragment shaders are used to perform the computation, a set of proxy polygons is required
     * to cover the output area of the image set which make up the output tensor.
     */
    virtual void setupNetworkPolygons(VAO *vao);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO *vertexArray_ = nullptr;      //!< Pointer to vertex-array object which maintains the VBO / IBO config
    VBO *vertexBuffer_ = nullptr;     //!< Pointer to VBO for the polygons used in the layer
    IBO *indexBuffer_ = nullptr;      //!< Pointer to IBO used for the polygons
    int downsample_[2] = {1, 1};      //!< Downsampling stride
    int poolSize_[2] = {1, 1};        //!< Pool size (x, y direction)
    bool equalAspect_ = true;         //!< Indicator that downsampling is isotropic (i.e. the same for x & y)
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
