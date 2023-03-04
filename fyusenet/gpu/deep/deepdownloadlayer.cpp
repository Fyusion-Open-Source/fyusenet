//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture -> CPU Buffer Download Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <algorithm>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deepdownloadlayer.h"
#include "deeptiler.h"
#include "../../gl/fbo.h"
#include "../../gl/pbopool.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc LayerBase::LayerBase()
 */
DeepDownloadLayer::DeepDownloadLayer(const UpDownLayerBuilder& builder, int layerNumber) :
    DeepLayerBase((const GPULayerBuilder &)builder,layerNumber), CPULayerInterface(), DownloadLayerInterface() {
    assert(builder.direction_ == UpDownLayerBuilder::DOWNLOAD);
    assert(inputPadding_ == outputPadding_);      // NOTE (mw) for now we do not allow padding change in this layer
    if (flags_ & LayerFlags::PRE_ACT_MASK) THROW_EXCEPTION_ARGS(FynException,"Activation on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
#ifdef FYUSENET_MULTITHREADING
    if (builder.callback_) userCallback_ = builder.callback_;
    async_ = builder.async_;
#endif
}

/**
 * @copydoc LayerBase::~LayerBase()
 */
DeepDownloadLayer::~DeepDownloadLayer() {
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepDownloadLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                tiler_->getInputTextureWidth(),tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4,TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,BufferSpec::FUNCTION_SOURCE));
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepDownloadLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, width_, height_,
                                BufferSpec::SINGLE32F, BufferSpec::SINGLE, BufferSpec::FLOAT,
                                BufferSpec::CPU_DEST, outputChannels_).device(BufferSpec::COMP_STOR_CPU).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/**
 * @copydoc cpu::CPULayerInterface::addOutputBuffer
 */
void DeepDownloadLayer::addOutputBuffer(CPUBuffer *buf, int port) {
    assert(buf);
    if (port != 0) THROW_EXCEPTION_ARGS(FynException, "Ports other than 0 are not supported");
    if (buf->shape().dataOrder() != CPUBufferShape::order::GPU_DEEP) {
        THROW_EXCEPTION_ARGS(FynException, "Buffers supplied to this layer must be in GPU_DEEP order");
    }
    outputs_.push_back(buf);
    assert(outputs_.size() <= 1);   // for now we only support one output buffer
    outputChanged_ = true;
}


/**
 * @copydoc cpu::CPULayerInterface::clearOutputBuffers
 */
void DeepDownloadLayer::clearOutputBuffers(int port) {
    assert(port == 0);
    outputs_.clear();
}


/**
 * @copydoc cpu::CPULayerInterface::hasOutputBuffer
 */
bool DeepDownloadLayer::hasOutputBuffer(int port) const {
    return (outputs_.size() > 0);
}


/**
 * @copydoc cpu::CPULayerInterface::getOutputBuffer
 */
CPUBuffer * DeepDownloadLayer::getOutputBuffer(int port) const {
    assert(port == 0);
    if ((int)outputs_.size() <= port) return nullptr;
    return outputs_.at(0);
}


/**
 * @copydoc LayerBase::setup
 */
void DeepDownloadLayer::setup() {
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void DeepDownloadLayer::forward(uint64_t sequence) {
    assert(outputs_.size() == 1);
    assert(numFBOs() == 1);
    // TODO (mw) implement optional rendering step here (for ReLU)
    ManagedPBO pbo = pboBlit();
#ifndef FYUSENET_MULTITHREADING
    if (true) {
#else
    if (!async_) {
#endif
        //-------------------------------------------------------------
        // Synchronous part, we still use a PBO here though there is no
        // advantage doing that. It just makes the code easier.
        //-------------------------------------------------------------
        outputs_[0]->readFromPBO(*pbo, CPUBufferShape::type::FLOAT32, sequence);
    } else {
#ifdef FYUSENET_MULTITHREADING
        THROW_EXCEPTION_ARGS(FynException, "Layer is not synchronous");
#endif
    }
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @copydoc DownloadLayerInterface::asyncForward
 */
void DeepDownloadLayer::asyncForward(uint64_t sequenceNo, const std::function<void(uint64_t)>& callback) {
    // TODO (mw) implement optional rendering step here (for ReLU)
    if (flags_ & LayerFlags::PRE_ACT_MASK) THROW_EXCEPTION_ARGS(FynException,"Activation on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
    if (!async_) THROW_EXCEPTION_ARGS(FynException, "Layer is not asynchronous");
    ManagedPBO pbo = pboBlit();
    //-------------------------------------------------------------
    // We issue a fence here and start a thread that waits for the
    // fence before reading out the PBO...
    //-------------------------------------------------------------
    GLsync sync = context().issueSync();
    asyncLock_.lock();
    auto thread = AsyncPool::getDerivedContextThread(context());
    threads_[sequenceNo] = thread;
    thread->setTask(std::bind(&DeepDownloadLayer::readoutPBO, this, thread, pbo, sync, sequenceNo, outputs_[0], callback));
    if (userCallback_) userCallback_(sequenceNo, outputs_[0], AsyncLayer::DOWNLOAD_COMMENCED);
    asyncLock_.unlock();
}
#endif

/**
 * @copydoc DownloadLayerInterface::wait
 */
void DeepDownloadLayer::wait(uint64_t sequenceNo) {
#ifdef FYUSENET_MULTITHREADING    
    if (async_) {
        asyncLock_.lock();
        auto it = threads_.find(sequenceNo);
        if (it != threads_.end()) {
            AsyncPool::GLThread thread = it->second;                   // make sure that we keep the refcount in the thread > 0
            asyncLock_.unlock();
            thread->wait();
        } else {
            asyncLock_.unlock();
        }
    }
#endif
}

/**
 * @brief Update output CPU buffer
 *
 * @param buf Pointer to CPUBuffer that is to replace the currently set buffer
 *
 * @param port Output port number, currently only 0 is accepted
 *
 * @note This function does not assume ownership over the supplied \p buf
 */
void DeepDownloadLayer::updateOutputBuffer(CPUBuffer *buf, int port) {
    assert(buf);
    if (port != 0) THROW_EXCEPTION_ARGS(FynException, "Ports other than 0 are not supported");
#ifdef FYUSENET_MULTITHREADING
    asyncLock_.lock();
#endif
    if (outputs_.size() != 1) THROW_EXCEPTION_ARGS(FynException, "No buffer position to be updated");
    outputs_[port] = buf;
    outputChanged_ = true;
#ifdef FYUSENET_MULTITHREADING
    asyncLock_.unlock();
#endif
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Blit texture content into %PBO
 *
 * @return ManagedPBO instance that wraps the %PBO in the operation
 *
 * This function blits the texture data into a %PBO which has sufficient capacity to hold the
 * content.
 */
ManagedPBO DeepDownloadLayer::pboBlit() {
    PBOPool *pool = context_.interface()->getReadPBOPool();
    assert(pool);
    int paddedwidth = viewport_[0];
    int paddedheight = viewport_[1];
    ManagedPBO pbo = pool->getAvailablePBO(paddedwidth, paddedheight, PIXEL_PACKING, bytesPerChan_);
    pbo->prepareForRead(paddedwidth * paddedheight * PIXEL_PACKING * bytesPerChan_);
    FBO *fbo = getFBO(0);
    fbo->bind();
    fbo->copyToPBO(*pbo, GL_FLOAT, PIXEL_PACKING, 0, true);
    fbo->unbind();
    if (async_) pbo.setPending();
    return pbo;
}

#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Perform readout of PBO memory buffer into destination CPUBuffer instance
 *
 * @param myThread Reference to GLThread that this function runs on
 * @param pbo Reference to a ManagedPBO instance which wraps the PBO to be read out
 * @param sync Handle of the OpenGL fence sync that indicates when the PBO is ready for readout
 * @param sequence Sequence number that refers to the content in the PBO to be read out
 * @param target Pointer to CPUBuffer where the data should be placed in
 * @param callback Callback function in the engine to pass notification about finished download
 *
 * @throw FynException in case the \p sync was not posted on the GL pipeline after less than 5s
 *
 * This function waits for the supplied \p sync to be issued on the GL pipeline in a background
 * thread (to be more precise, it is invoked in the background thread already). Once the sync has
 * been received, the \p pbo will be mapped into memory and the data will be copied to the buffer(s)
 * in #outputs_. After reading the data, two callbacks will be invoked:
 *   - \p callback which notifies the engine that the PBO has been read
 *   - #userCallback_ which notifies the API user that the PBO has been read
 *
 * The latter callback is optional and is supplied via the UpDownLayerBuilder.
 *
 * @see UpDownLayerBuilder, Engine::asyncDownloadDone
 */
void DeepDownloadLayer::readoutPBO(AsyncPool::GLThread& myThread, opengl::ManagedPBO& pbo, GLsync sync, uint64_t sequence, cpu::CPUBuffer * target, const std::function<void(uint64_t)> & callback) {
    using namespace opengl;
    const GfxContextLink & ctx = myThread.context();
    bool rc = ctx.waitClientSync(sync, 5000000000);        // wait 5s max  (TODO (mw) configurable timeout)
    if (!rc) THROW_EXCEPTION_ARGS(FynException, "Cannot read out texture within 5s for sequence %ld", sequence);
    ctx.removeSync(sync);
    target->readFromPBO(*pbo, CPUBufferShape::type::FLOAT32, sequence);
    pbo.clearPending();
    if (callback) callback(sequence);
    if (userCallback_) userCallback_(sequence, target, AsyncLayer::DOWNLOAD_DONE);
    asyncLock_.lock();
    auto it = threads_.find(sequence);
    assert(it != threads_.end());
    threads_.erase(it);
    asyncLock_.unlock();
}
#endif


/**
 * @copydoc GPULayerBase::setupFBOs()
 */
void DeepDownloadLayer::setupFBOs() {
    assert(inputChannels_ == outputChannels_);
    if (flags_ & LayerFlags::PRE_RELU) THROW_EXCEPTION_ARGS(FynException,"ReLU on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
    // we can directly connect the input textures to the FBOs here for now, as we
    // currently only support downloading float data without any flags
    FBO * fbo = new FBO(context_, viewport_[0], viewport_[1], inputTextures_.at(0));
    framebuffers_.push_back(fbo);
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::updateFBOs()
 */
void DeepDownloadLayer::updateFBOs() {
    outputChanged_ = false;
}

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
