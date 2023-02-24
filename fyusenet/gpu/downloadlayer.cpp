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

#include "downloadlayer.h"
#include "../common/fynexception.h"
#include "../gl/fbo.h"
#include "../cpu/cpubuffer.h"
#include "../base/bufferspec.h"
#include "../gl/pbo.h"
#include "../gl/pbopool.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
DownloadLayer::DownloadLayer(const UpDownLayerBuilder& builder, int layerNumber) :
    GPULayerBase((const GPULayerBuilder &)builder,layerNumber), CPULayerInterface(), DownloadLayerInterface() {
    assert(inputPadding_ == outputPadding_);
    assert(builder.direction_ == UpDownLayerBuilder::DOWNLOAD);
    assert(LayerBase::inputPadding_ == LayerBase::outputPadding_);      // NOTE (mw) for now we do not allow padding change in this layer
#ifdef FYUSENET_MULTITHREADING
    if (builder.callback_) userCallback_ = builder.callback_;
    async_ = builder.async_;
#endif
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
DownloadLayer::~DownloadLayer() {
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DownloadLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    int channel = 0;
    int rem = inputChannels_;
    while (rem > 0) {
        result.push_back(BufferSpec(channel++, 0,
                                    width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::FUNCTION_SOURCE));
        rem -= PIXEL_PACKING;
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DownloadLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                width_ + 2*outputPadding_, height_ + 2*outputPadding_,
                                BufferSpec::SINGLE32F, BufferSpec::SINGLE, BufferSpec::FLOAT, BufferSpec::CPU_DEST,
                                outputChannels_).interpolation(BufferSpec::NEAREST).device(BufferSpec::COMP_STOR_CPU).dataOrder(BufferSpec::order::GPU_SHALLOW));
    return result;
}


/**
 * @copydoc cpu::CPULayerInterface::clearOutputBuffers
 */
void DownloadLayer::clearOutputBuffers(int port) {
    assert(port == 0);
    outputs_.clear();
}


/**
 * @copydoc LayerBase::setup
 */
void DownloadLayer::setup() {
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void DownloadLayer::forward(uint64_t sequence) {
    assert(outputs_.size() == 1);
    // TODO (mw) implement optional rendering step here (for ReLU)
    if (flags_ & LayerFlags::PRE_ACT_MASK) THROW_EXCEPTION_ARGS(FynException,"Activation on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
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
void DownloadLayer::asyncForward(uint64_t sequenceNo, const std::function<void (uint64_t)> &callback) {
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
    thread->setTask(std::bind(&DownloadLayer::readoutPBO, this, thread, pbo, sync, sequenceNo, outputs_[0], callback));
    if (userCallback_) userCallback_(sequenceNo, outputs_[0], AsyncLayer::DOWNLOAD_COMMENCED);
    asyncLock_.unlock();
}
#endif


/**
 * @copydoc DownloadLayerInterface::wait
 */
void DownloadLayer::wait(uint64_t sequenceNo) {
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
 * @copydoc GPULayerBase::updateFBOs
 */
void DownloadLayer::updateFBOs() {
    outputChanged_ = false;
}


/**
 * @copydoc cpu::CPULayerInterface::addOutputBuffer
 */
void DownloadLayer::addOutputBuffer(CPUBuffer *buf, int port) {
    assert(buf);
    if (port != 0) THROW_EXCEPTION_ARGS(FynException, "Ports other than 0 are not supported");
    if (outputs_.size() > 0) THROW_EXCEPTION_ARGS(FynException,"Only one output buffer is supported for this layer type");
    outputs_.push_back(buf);   
    outputChanged_ = true;
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
void DownloadLayer::updateOutputBuffer(CPUBuffer *buf, int port) {
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


/**
 * @copydoc cpu::CPULayerInterface::hasOutputBuffer
 */
bool DownloadLayer::hasOutputBuffer(int port) const {
    assert(port >= 0);
    if ((int)outputs_.size() <= port) return false;
    return (outputs_.at(port) != nullptr);
}


/**
 * @copydoc cpu::CPULayerInterface::getOutputBuffer
 */
CPUBuffer * DownloadLayer::getOutputBuffer(int port) const {
    assert(port >= 0);
    if ((int)outputs_.size() <= port) return nullptr;
    return outputs_.at(port);
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
ManagedPBO DownloadLayer::pboBlit() {
    // a lot of PBO binds/unbinds here, maybe consolidate when there is time
    PBOPool *pool = context_.interface()->getReadPBOPool();
    assert(pool);
    int paddedwidth = width_ + 2*inputPadding_;
    int paddedheight = height_ + 2*inputPadding_;
    int paddedchannels = LayerBase::PIXEL_PACKING * ((outputChannels_ + LayerBase::PIXEL_PACKING - 1) / LayerBase::PIXEL_PACKING);
    ManagedPBO pbo = pool->getAvailablePBO(paddedwidth, paddedheight, paddedchannels, bytesPerChan_);
    pbo->prepareForRead(paddedwidth * paddedheight * paddedchannels * bytesPerChan_);
    int readchans = 0;
    pbo->bind(GL_PIXEL_PACK_BUFFER);
    for (int fb = 0 ; fb < numFBOs(); fb++ ) {
        // NOTE (mw) we assume that the FBOs are putting out all 4 channels
        size_t offset = readchans * paddedwidth * paddedheight * bytesPerChan_;
        FBO *fbo = getFBO(fb);
        fbo->bind();
        int chans = LayerBase::PIXEL_PACKING * fbo->numAttachments();
        fbo->copyToPBO(*pbo, GL_FLOAT, LayerBase::PIXEL_PACKING, offset);
        fbo->unbind();
        readchans += chans;
    }
    pbo->unbind(GL_PIXEL_PACK_BUFFER);
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
void DownloadLayer::readoutPBO(AsyncPool::GLThread& myThread, opengl::ManagedPBO& pbo, GLsync sync, uint64_t sequence, cpu::CPUBuffer * target, const std::function<void(uint64_t)> & callback) {
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
 * @copydoc GPULayerBase::setupFBOs
 */
void DownloadLayer::setupFBOs() {
    assert(inputChannels_ == outputChannels_);
    if (flags_ & LayerFlags::PRE_ACT_MASK) THROW_EXCEPTION_ARGS(FynException,"Activation on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
    // we can directly connect the input textures to the FBOs here for now, as we
    // currently only support downloading float data without any flags
    int numpasses = (inputTextures_.size()  + maxRenderTargets_ - 1) / maxRenderTargets_;
    int texoffset = 0;
    for (int pass=0; pass < numpasses; pass++) {
        FBO * fbo = new FBO(context_, viewport_[0], viewport_[1], inputTextures_.at(texoffset++));
        int pack = 1;
        while ((pack < maxRenderTargets_) && (texoffset < (int)inputTextures_.size())) {
            fbo->addTexture(GL_COLOR_ATTACHMENT0+pack,inputTextures_.at(texoffset++), GL_TEXTURE_2D);
            pack++;
        }
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_ = false;
}



} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
