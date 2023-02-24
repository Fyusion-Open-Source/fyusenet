//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture -> CPU Buffer Download Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <vector>
#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../downloadinterface.h"
#include "../../base/asynclayerinterface.h"
#include "deepfunctionlayer.h"
#include "../../base/bufferspec.h"
#include "../../gl/gl_sys.h"
#ifdef FYUSENET_MULTITHREADING
#include "../../gl/asyncpool.h"
#endif
#include "../../gl/managedpbo.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/uniformstate.h"
#include "../../cpu/cpubuffer.h"
#include "../../cpu/cpulayerinterface.h"
#include "../updownlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {

namespace opengl {
  class PBO;
}

namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Download layer from GPU to CPU for deep tensor data
 *
 * This layer performs a "download" of GPU tensor data to the CPU. By download we mean reading back
 * data from a texture into CPU memory via an FBO. For performance reasons, this layer supports
 * asynchronous operation. The reason is that the download has to use the infamous
 * <a href="https://www.khronos.org/opengl/wiki/GLAPI/glReadPixels">glReadPixels()</a> function, which
 * introduces a set of problems, most notably the high latency for this call. As OpenGL pipelines are
 * not in sync with CPU code, obtaining an image from the pipeline might take a long time and the
 * \c glReadPixels() call basically stalls until the GPU part is done with the rendering at this
 * point.
 *
 * The asynchronous operating mode of this layer only triggers the download by reading the texture
 * data into a PBO instance and issues a sync on the GL pipeline before using a background thread to
 * wait for the sync and then map the PBO into CPU memory for the readout. The background thread
 * will notify the Engine via a callback and will then also call an optional user-supplied callback.
 *
 * To allow for flexibility on the output buffers for async operation, this layer does not control
 * the output buffers itself, it is rather the individual network implementation that must take care
 * of this. The suggested pattern is to provide a callback to the associated builder which takes
 * care of the buffer swaps. An implementation of a callback could look like this:
 *
 * @code
 * void callback(uint64_t seqNo, CPUBuffer *buffer, AsyncLayer::state state) {
 *    ...
 *    if (state == AsyncLayer::state::DOWNLOAD_COMMENCED) {
 *        // an async download has been launched in a background thread,
 *        // the output buffer of the layer can now be safely swapped
 *    } else
 *    if (state == AsyncLayer::state::DOWNLOAD_DONE) {
 *        // an async download has completed copying the data into the
 *        // target buffer which can now be used
 *    }
 *    ...
 * }
 * @endcode
 *
 * The code in the callback should be considered time-critical, so if complex operations do need
 * to be performed on the buffer, those should be relayed to a different thread if performance
 * is of the essence.
 *
 * @see Engine::asyncDownloadDone, UpDownLayerBuilder
 */
class DeepDownloadLayer : public DeepLayerBase, public cpu::CPULayerInterface, public DownloadLayerInterface, public AsyncLayer {
    friend class fyusion::fyusenet::Engine;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepDownloadLayer(const UpDownLayerBuilder& builder, int layerNo);
    virtual ~DeepDownloadLayer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void setup() override;
    virtual void forward(uint64_t sequence) override;
#ifdef FYUSENET_MULTITHREADING
    virtual void asyncForward(uint64_t sequence, const std::function<void(uint64_t)>& callback) override;
#endif
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    virtual CPUBuffer * getOutputBuffer(int port=0) const override;
    virtual void addOutputBuffer(cpu::CPUBuffer *buf, int port=0) override;
    virtual bool hasOutputBuffer(int port=0) const override;    
    virtual void clearOutputBuffers(int port = -1) override;
    virtual void wait(uint64_t sequenceNo) override;
    void updateOutputBuffer(CPUBuffer *buf, int port=0);

    /**
     * @brief Check if download layer is asynchronous
     *
     * @retval true if layer is asynchronous
     * @retval false otherwise
     */
    virtual bool isAsync() const override {
        return async_;
    }

    /**
     * @brief Get input buffer
     *
     * @param port Input port to retrieve the buffer for
     *
     * @return throws an exception (always)
     *
     * @throw FynException always
     */
    virtual CPUBuffer * getInputBuffer(int port=0) const override {
        THROW_EXCEPTION_ARGS(FynException,"Input buffers are not supported for this layer type");
    }


    /**
     * @brief Clear/reset input buffers for this layer (unsupported)
     *
     * @param port Output port to clear the buffers from, or -1 to clear \e all ports
     *
     * @throws FynException always
     *
     * As this layer does not support input buffers, this function will always throw an exception
     */
    virtual void clearInputBuffers(int port = -1) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for download layer");
    }

    /**
     * @copydoc cpu::CPULayerInterface::setInputBuffer
     */
    virtual void setInputBuffer(CPUBuffer * buf,int port) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for download layer");
    }

    /**
     * @copydoc cpu::CPULayerInterface::setResidualBuffer
     */
    virtual void setResidualBuffer(CPUBuffer * buf) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for download layer");
    }
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupFBOs() override;
    virtual void updateFBOs() override;
#ifdef FYUSENET_MULTITHREADING
    void readoutPBO(AsyncPool::GLThread& myThread, opengl::ManagedPBO& pbo, GLsync sync, uint64_t sequence, cpu::CPUBuffer *target, const std::function<void(uint64_t)> & callback);
#endif
    ManagedPBO pboBlit();
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    uint8_t bytesPerChan_ = 4;                        //!< Number of bytes per channel (defaults to 4 bytes for a single-precision floating point number)
    bool async_ = false;                              //!< Indicator if this is an asynchronous download layer
    std::vector<cpu::CPUBuffer *> outputs_;           //!< Output CPU buffer(s)
    /**
     * Optional user callback function for asynchronous operation
     */
    std::function<void(uint64_t, cpu::CPUBuffer *, AsyncLayer::state)> userCallback_;
#ifdef FYUSENET_MULTITHREADING
    std::recursive_mutex asyncLock_;                  //!< Serializes access to #threads_ and #outputs_
    /**
     * Currently running download threads indexed by sequence numbers
     *
     * @see #asyncLock_
     */
    std::unordered_map<uint64_t, opengl::AsyncPool::GLThread> threads_;
#endif
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
