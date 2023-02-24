//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture Upload Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <atomic>
#include <functional>
#include <condition_variable>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../base/asynclayerinterface.h"
#include "../common/fynexception.h"
#include "updownlayerbuilder.h"
#include "gpulayerbase.h"
#include "../cpu/cpulayerinterface.h"
#include "../cpu/cpubuffer.h"
#include "../gl/pbopool.h"
#include "../gl/managedpbo.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Layer to upload CPU data to the GPU
 *
 *
 * In order to get data from the CPU to the GPU and vice versa it needs to be uploaded/downloaded
 * to/from the GPU. In terms of OpenGL, the upload is usually done via a call to
 * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml">glTexImage2D</a>
 * Those calls introduce significant delay and OpenGL offers a few workarounds to alleviate the
 * associated time penalty. Using asynchronous upload and download in conjunction with
 * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glFenceSync.xhtml">fences</a>,
 * the throughput in the processing pipeline is maximized. The latency however will not be
 * reduced significantly.
 *
 * Due to the primary goal of FyuseNet being a GPU inference engine, this class (currently) does
 * not offer any convenience in regards to buffer shapes. In particular that means that if you want
 * to upload a buffer, the data in the buffer must be formatted appropriately such that a batch of
 * 4 channels is always aggregated as a single element (think of it as RGBA pixels, which they are).
 * Data with less than 4 channels must be aggregated as either 1,2 or 3-channel elements, following
 * a simple convention as: red-channel data, red-green data and RGB data.
 *
 * On asynchronous uploads it is important to keep track of when the input buffer may be changed.
 * The UpDownLayerBuilder offers to add a callback function, which will be invoked by the uploading
 * thread \e after the buffer contents of the original input buffer have been copied and it is safe
 * to change the input buffer. Failure to comply with the signalling may lead to inconsistencies
 * in the uploaded data.
 *
 * In order to make sure not to overwrite %PBO buffer data \e before it was actually set up as
 * texture (due to asynchronicity between the CPU and the GPU), the individual texture set
 * (currently 2) of this layer has to be "unlocked" before the next (asynchronous) upload on the
 * same set can start. The unlocking must happen \e after \e all layers that consume the texture(s)
 * written by this layer have read the data and written their own output. The only way to  ensure
 * that is by using appopriate fences which is handled by the Engine.
 *
 * @see Engine::waitForUploadFence(), Engine::execute()
 */
class UploadLayer : public GPULayerBase, public cpu::CPULayerInterface, public AsyncLayer {
    // NOTE (mw) a lot of assumptions in the code rely on this number to be 2, refactor the code if you change this to higher numbers
    constexpr static int ASYNC_BUFFERS = 2;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    UploadLayer(const UpDownLayerBuilder & builder, int layerNo);
    virtual ~UploadLayer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void setup() override;
    virtual void cleanup() override;
    virtual void setInputBuffer(CPUBuffer * buf, int port) override;
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    virtual void forward(uint64_t sequence=0) override;
#ifdef FYUSENET_MULTITHREADING
    bool asyncForward(uint64_t sequence, const std::function<void(uint64_t)>& engineCallback);
    void swapOutputTextures(uint64_t sequence);
    bool isLocked() const;
    void unlock(uint64_t sequenceNo);
#endif
    virtual void updateFBOs() override;
    virtual void clearInputBuffers(int port = -1) override;
    virtual void addOutputTexture(GLuint textureID, int channelIndex, int shadowIndex=0) override;

    /**
     * @brief Get input buffer
     *
     * @param port Input port to retrieve the buffer for
     *
     * @return Input buffer assigned to specified port
     */
    virtual CPUBuffer * getInputBuffer(int port=0) const override {
        assert(port == 0);
#ifdef FYUSENET_MULTITHREADING
        std::lock_guard<std::mutex> guard(asyncLock_);
#endif
        return input_;
    }

    /**
     * @brief Check if upload layer is asynchronous
     *
     * @retval true if layer is asynchronous
     * @retval false otherwise
     */
    virtual bool isAsync() const override {
#ifdef FYUSENET_MULTITHREADING
        return async_;
#else
        return false;
#endif
    }

    /**
     * @brief Clear/reset output buffers for this layer (unsupported)
     *
     * @param port Output port to clear the buffers from, or -1 to clear \e all ports
     *
     * @throws FynException always
     *
     * As this layer does not support output buffers, this function will always throw an exception
     */
    virtual void clearOutputBuffers(int port = -1) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for upload layer");
    }

    /**
     * @brief Check if output buffer at specified port has been set
     *
     * @param port Output port number
     *
     * @retval false
     *
     * This function always returns \c false, since CPU output buffers are not supported by
     * texture uploads.
     */
    virtual bool hasOutputBuffer(int port=0) const override {
        return false;
    }

    /**
     * @brief Get output buffer (returns \c nullptr)
     *
     * @param port Output port to retrieve the buffer for
     *
     * @return \c nullptr
     *
     * This function is not supported for this layer type and hence always returns a \c nullptr
     */
    virtual CPUBuffer * getOutputBuffer(int port=0) const override {
        return nullptr;
    }


    /**
     * @brief Add output buffer (unsupported)
     *
     * @param buf Pointer to CPU buffer to write to
     * @param port Output port (defaults to 0)
     *
     * This function throws an exception because it is not supported for this layer type
     */
    virtual void addOutputBuffer(CPUBuffer *buf, int port=0) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for upload layer");
    }

    /**
    * @brief Add residual buffer (unsupported)
    *
    * This function throws an exception because it is not supported for this layer type
    */
    virtual void setResidualBuffer(CPUBuffer * buf) override {
        THROW_EXCEPTION_ARGS(FynException,"Not supported for upload layer");
    }
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void syncUpload();
#ifdef FYUSENET_MULTITHREADING
    bool asyncUpload(uint64_t sequence, const std::function<void(uint64_t)> & callback);
    void asyncUploadTask(opengl::ManagedPBO& pbo, const void *srcData, uint64_t sequence, CPUBuffer * buffer, int texIdx,  const std::function<void(uint64_t)> & callback);
    void updateDependencies(const std::vector<GLuint>& textures) const;
#endif
    virtual void setupFBOs() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    BufferSpec::dtype dataType_;                    //!< Data type for this layer (currently only bytes and 32-bit floats are supported)
    CPUBuffer * input_ = nullptr;                   //!< Pointer to assigned input CPU buffer
    bool async_ = false;                            //!< Synchronous/Asynchronous upload mode toggle
    uint8_t bytesPerChan_ = 0;                      //!< Bytes per channel
#ifdef FYUSENET_MULTITHREADING
    mutable std::mutex asyncLock_;                  //!< Locks access to members used for asynchronous uploads
    uint64_t inFlight_[ASYNC_BUFFERS];              //!< Stores sequence numbers of in-flight uploads, the index in the array relates to the texture set
    int locked_ = 0;                                //!< Number of locked texture sets, also see #asyncLock_

    /**
     * Optional user callback on completion of an asynchronous upload
     */
    std::function<void(uint64_t, cpu::CPUBuffer *, AsyncLayer::state)> userCallback_;

    /**
     * Multi-buffer shadow texture IDs
     */
    std::vector<GLuint> shadowTextures_[ASYNC_BUFFERS-1];
#endif
};


}  // gpu namespace
}  // fyusenet namespace
}  // fyusion namespace


// vim: set expandtab ts=4 sw=4:

