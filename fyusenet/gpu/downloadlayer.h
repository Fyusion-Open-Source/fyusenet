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
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "downloadinterface.h"
#include "../base/asynclayerinterface.h"
#include "gpulayerbase.h"
#include "../cpu/cpulayerinterface.h"
#include "updownlayerbuilder.h"
#include "../gl/gl_sys.h"
#include "../gl/managedpbo.h"
#ifdef FYUSENET_MULTITHREADING
#include "../gl/asyncpool.h"
#endif
#include "../common/fynexception.h"
#include "../cpu/cpubuffer.h"

//------------------------------------- Public Declarations ----------------------------------------

using fyusion::fyusenet::cpu::CPUBuffer;
using fyusion::fyusenet::BufferShape;

namespace fyusion {

namespace opengl {
  class PBO;
}

namespace fyusenet::gpu {

/**
 * @brief Download layer from GPU to CPU for shallow tensor data
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
class DownloadLayer : public GPULayerBase, public cpu::CPULayerInterface, public DownloadLayerInterface, public AsyncLayer {
    friend class fyusion::fyusenet::Engine;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DownloadLayer(const UpDownLayerBuilder& builder, int layerNo);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup() override;
    void forward(uint64_t sequenceNo, StateToken * state) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
#ifdef FYUSENET_MULTITHREADING
    virtual void asyncForward(uint64_t sequenceNo, StateToken *token, const std::function<void(uint64_t)>& callback) override;
#endif
    void updateFBOs() override;
    void addCPUOutputBuffer(CPUBuffer *buf, int port=0) override;
    [[nodiscard]] bool hasCPUOutputBuffer(int port=0) const override;
    [[nodiscard]] CPUBuffer * getCPUOutputBuffer(int port=0) const override;
    void clearCPUOutputBuffers(int port = -1) override;
    void wait(uint64_t sequenceNo) override;
    void updateOutputBuffer(CPUBuffer *buf, int port=0);
    [[nodiscard]] BufferShape getOutputShape(int port) const override;

    /**
     * @brief Check if download layer is asynchronous
     *
     * @retval true if layer is asynchronous
     * @retval false otherwise
     */
    [[nodiscard]] bool isAsync() const override {
        return async_;
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
    void clearCPUInputBuffers(int port = -1) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for download layer");
    }

    /**
     * @copydoc cpu::CPULayerInterface::setCPUInputBuffer
     */
    void setCPUInputBuffer(CPUBuffer * buf, int port) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for download layer");
    }

    /**
     * @copydoc cpu::CPULayerInterface::setCPUResidualBuffer
     */
    void setCPUResidualBuffer(CPUBuffer * buf) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for download layer");
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
    [[nodiscard]] CPUBuffer * getCPUInputBuffer(int port=0) const override {
        THROW_EXCEPTION_ARGS(FynException,"Input buffers are not supported for this layer type");
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupFBOs() override;
    ManagedPBO pboBlit();
    static BufferSpec::sizedformat bufferFormat(BufferSpec::dtype type, int packing);
    static bool isInt(BufferSpec::dtype type);
#ifdef FYUSENET_MULTITHREADING
    void readoutPBO(opengl::AsyncPool::GLThread& myThread, opengl::ManagedPBO& pbo, GLsync sync, uint64_t sequence, cpu::CPUBuffer * target, const std::function<void(uint64_t)> & callback);
#endif
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int bytesPerChan_ = 4;                                //!< Number of bytes per channel (defaults to 4 bytes for a single-precision floating point number)
    bool async_ = false;                                  //!< Indicator if this is an asynchronous download layer
    int maxRenderTargets_ = 1;                            //! Maximum number of render targets for a single run
    int maxSequence_ = 0;
    int sequenceLen_ = 0;
    int chanPacking_ = PIXEL_PACKING;                     //!< Element packing mode for texture data
    std::vector<CPUBuffer *> outputs_;                    //!< Output CPU buffer(s)
    BufferSpec::dtype dataType_ = TEXTURE_TYPE_DEFAULT;   //!< Input data type for this layer

    /**
     * Optional user callback function for asynchronous operation
     */
    std::function<void(uint64_t, cpu::CPUBuffer *, AsyncLayer::state)> userCallback_;
#ifdef FYUSENET_MULTITHREADING
    std::recursive_mutex asyncLock_;                    //!< Serializes access to #threads_ and #outputs_
    /**
     * Currently running download threads indexed by sequence numbers
     *
     * @see #asyncLock_
     */
    std::unordered_map<uint64_t, opengl::AsyncPool::GLThread> threads_;
#endif
};

} // fyusenet::gpu namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
