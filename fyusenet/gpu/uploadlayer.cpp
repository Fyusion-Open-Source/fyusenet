//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture Upload Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "uploadlayer.h"
#ifdef FYUSENET_MULTITHREADING
#include "../gl/asyncpool.h"
#endif

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
UploadLayer::UploadLayer(const UpDownLayerBuilder& builder, int layerNumber) :
    GPULayerBase((const GPULayerBuilder& )builder,layerNumber), CPULayerInterface() {
    assert(builder.direction_ == UpDownLayerBuilder::UPLOAD);
    assert(LayerBase::inputPadding_ == LayerBase::outputPadding_);      // NOTE (mw) for now we do not allow padding change in this layer
    assert(inputChannels_ == outputChannels_);
    // TODO (mw) user callback support
#ifdef FYUSENET_MULTITHREADING
    memset(inFlight_, 0, ASYNC_BUFFERS * sizeof(uint64_t));
    async_ = builder.async_;
    userCallback_ = builder.callback_;
#endif
    dataType_ = builder.dataType_;
    switch (dataType_) {
        case BufferSpec::FLOAT16:
            // intentional fallthrough (we cannot upload half-float data and let the driver do the conversion
        case BufferSpec::FLOAT:
            // intentional fallthrough
        case BufferSpec::UINT:
            bytesPerChan_ = 4;
            break;
        case BufferSpec::UBYTE:
            bytesPerChan_ = 1;
            break;
    }
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
UploadLayer::~UploadLayer() {
}


/**
 * @copydoc LayerBase::setup
 */
void UploadLayer::setup() {
    // empty on purpose
}


/**
 * @copydoc LayerBase::cleanup
 */
void UploadLayer::cleanup() {
    // empty on purpose
}


/**
 * @copydoc LayerBase::forward
 */
void UploadLayer::forward(uint64_t sequence) {
    if (!input_) THROW_EXCEPTION_ARGS(FynException,"No input buffer set for upload");
    if ((dataType_ != BufferSpec::FLOAT) && (dataType_ != BufferSpec::UBYTE)) {
        THROW_EXCEPTION_ARGS(FynException, "Currently only 32-bit float and 8-bit uint are supported");
    }
#ifndef FYUSENET_MULTITHREADING
    if (true) {
#else
    if (!async_) {
#endif
        syncUpload();
    } else {        
#ifdef FYUSENET_MULTITHREADING
        THROW_EXCEPTION_ARGS(FynException, "Please use asyncForward() for asynchronous upload layers (%s)", getName().c_str());
#endif
    }
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Perform asynchronous computation
 *
 * @param sequence Sequence number for the computation
 * @param engineCallback Callback function for the inference engine which is invoked once the
 *                       upload has completely been pushed to the GL pipeline
 *
 * @retval true if asynchronous operation could commence
 * @retval false otherwise
 *
 * This function performs the equivalent of the forward() function, just in an asynchronous
 * fashion. The function returns immediately and all further state communication is done via
 * callbacks.
 *
 * @see asyncUpload()
 */
bool UploadLayer::asyncForward(uint64_t sequence, const std::function<void (uint64_t)> &engineCallback) {
    if (!input_) THROW_EXCEPTION_ARGS(FynException,"No input buffer set for upload");
    if ((dataType_ != BufferSpec::FLOAT) && (dataType_ != BufferSpec::UBYTE)) {
        THROW_EXCEPTION_ARGS(FynException, "Currently only 32-bit float and 8-bit uint are supported");
    }
    if (!async_) THROW_EXCEPTION_ARGS(FynException, "Layer %s is not asynchronous", getName().c_str());
    return asyncUpload(sequence, engineCallback);
}
#endif



/**
 * @copydoc GPULayerBase::updateFBOs
 */
void UploadLayer::updateFBOs() {
    outputChanged_ = false;
}


/**
 * @brief Add a single input buffer to specified port
 *
 * @param buf Pointer to CPU-side buffer that should be attached to this layer
 *
 * @param port Port number to connect buffer to, currently only port 0 is supported
 *
 * This function adds the supplied \p buf as input buffer to port 0 (the only valid port for this
 * layer). For data that has more than 4 channels, this function assumes that the supplied
 * buffer is formatted appropriately in \e shallow GPU format, that is that 4 channels are always
 * aggregated as a single element (think of it as an RGBA pixel, which it is). Data with less than
 * 4 channels must be aggregated as either 1,2 or 3-channel elements, following a simple convention
 * as red-only data, red-green data and RGB data.
 *
 * @see getRequiredInputBuffers
 *
 * @throws FynException in case the buffer is changed while an asynchronous upload is in
 *         progress
 *
 * @note This class does not take ownership over the supplied buffer, it is up to the caller to
 *       maintain its life-cycle.
 *
 * @warning Setting / updating the input buffer in asynchronous uploads is a bit tricky. Make
 *          sure that you only update the buffer and/or re-use the old buffer after it has been
 *          copied internally, which is usually signalled by this layer calling the callback
 *          function that was supplied in the builder.
 */
void UploadLayer::setInputBuffer(cpu::CPUBuffer *buf, int port) {
    assert(port == 0);
    assert(buf->shape().dataOrder() == BufferSpec::order::GPU_SHALLOW);
    if (isAsync()) {
#ifdef FYUSENET_MULTITHREADING
        std::lock_guard<std::mutex> lck(asyncLock_);
#endif
        input_ = buf;
    } else {
        input_ = buf;
    }
}



/**
 * @brief Register output texture with this layer
 *
 * @param textureID Raw OpenGL texture handle to be added to the list of output textures
 *
 * @param channelIndex Index which is based on output channel
 *
 * @param shadowIndex Optional index that adds a multiple textures to the same port/channelindex
 *                    for multi-buffering.
 *
 * This function adds a texture to the output texture list at the provided \p channelIndex location.
 * Opposed to the input, layers currently only have one output port, but may be extended to support
 * multiple output ports later (or never).
 *
 * @post #outputChanged_ is set to \c true to indicate that some parts may have to be reinitialized
 *
 * @note This class does not take ownership over the supplied texture, it is up to the caller to
 *       maintain its life-cycle.
 *       Unlike the input, layers currently only have one output port, but may be extended to
 *       support multiple output ports later (or never). If a layer has more than one output port
 *       in the future, each port may consist of more than one texture. The \p channelIndex specifies
 *       a flattened offset into this list. For example, assume that a layer has 2 output ports,
 *       where the first port has 24 channels and the second port has 32 channels. This equals to 6
 *       textures on the first port and 8 textures on the second port (each texture aggregates 4
 *       channels). A channel index of 4 will therefore be channels 16 to 19 (inclusive) of the
 *       first port and a channel index of 12 will be channels 24..27 (inclusive) of the second port.
 *
 * @throws FynException if invalid parameters are supplied
 *
 * @see BufferManager::connectLayers, BufferManager::connectGPULayers, getOutputTexture
 */
void UploadLayer::addOutputTexture(GLuint textureID, int channelIndex, int shadowIndex) {
#ifndef FYUSENET_MULTITHREADING
    if (shadowIndex != 0) THROW_EXCEPTION_ARGS(FynException,"Illegal shadow index %d supplied, no multithreading support", shadowIndex);
#else
    if (shadowIndex != 0) {
        if (shadowIndex >= ASYNC_BUFFERS) THROW_EXCEPTION_ARGS(FynException, "Shadow index %d out of bounds", shadowIndex);
        while ((int)shadowTextures_[shadowIndex-1].size() < channelIndex) shadowTextures_[shadowIndex-1].push_back(0);
        if (channelIndex == (int)shadowTextures_[shadowIndex-1].size()) shadowTextures_[shadowIndex-1].push_back(textureID);
        else shadowTextures_[shadowIndex-1][channelIndex] = textureID;
        return;
    }
#endif
    GPULayerBase::addOutputTexture(textureID, channelIndex, shadowIndex);
}

/**
 * @copydoc cpu::CPULayerInterface::clearInputBuffers
 */
void UploadLayer::clearInputBuffers(int port) {
    assert(port == 0);
    input_ = nullptr;
}



/**
 * @brief Obtain buffer specifiers that are required as input for this layer
 *
 * @return Vector of buffer specifiers that specify the format for each required buffer
 *
 * This function returns a list of buffer specifiers for the CPU buffer(s) required to
 * upload the contained data to the GPU. Due to the primary goal of FyuseNet being a GPU
 * inference engine, this function (currently) does not offer any convenience in regards to
 * buffer shapes. In particular that means that if you want to upload a buffer that has more than
 * 4 channels, the data will have to be arranged in \e shallow GPU order, which in cases of
 * channels >=4 aggregates 4 channels in a single element (think of it as RGBA, which it is).
 *
 * @see BufferSpec
 */
std::vector<BufferSpec> UploadLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, width_ + 2*outputPadding_, height_ + 2*outputPadding_,
                                BufferSpec::SINGLE32F, BufferSpec::SINGLE, dataType_, BufferSpec::CPU_SOURCE,
                                inputChannels_).device(BufferSpec::COMP_STOR_CPU).dataOrder(BufferSpec::order::GPU_SHALLOW));
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> UploadLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    int rem = inputChannels_;
    int channelidx = 0;
    // FIXME (mw) this function will create problems when uploading channel data that is >4 and not a multiple of 4
    if (rem < PIXEL_PACKING) {
        auto format = BufferSpec::formatByChannels(inputChannels_, TEXTURE_TYPE_DEFAULT);
        result.push_back(BufferSpec(channelidx++, 0, width_+2*inputPadding_, height_+2*inputPadding_,
                                    format.first, format.second, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::GPU_DEST).async(async_).multi((async_) ? ASYNC_BUFFERS : 1));
    } else {
        while (rem > 0) {
            result.push_back(BufferSpec(channelidx++, 0,
                                        width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                                        TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                        BufferSpec::GPU_DEST).async(async_).multi((async_) ? ASYNC_BUFFERS : 1));
            rem -= LayerBase::PIXEL_PACKING;           // we don't care if we underflow here
        }
    }
    return result;
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Check if (asynchronous) upload layer is locked
 *
 * @retval true if layer is locked and cannot process another upload (yet)
 * @retval false if layer is able to process another asynchronous upload
 */
bool UploadLayer::isLocked() const {
    if (!async_) return false;
    else {
        std::lock_guard<std::mutex> lck(asyncLock_);
        return (locked_ >= ASYNC_BUFFERS);
    }
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Unlock textures that were used in the supplied sequence ID
 *
 * @param sequenceNo Sequence number of the upload run
 *
 * This function "unlocks" the texture(s) used for the upload that was performed at the supplied
 * \p sequenceNo. In order to increase throughput, the upload layer might use more than one set
 * of textures to perform the uploads and the \p sequenceNo is used to identify which set of
 * textures can be re-used.
 */
void UploadLayer::unlock(uint64_t sequenceNo) {
    if (async_) {
        std::lock_guard<std::mutex> lck(asyncLock_);
        int initial = locked_;
        for (int i=0; i < ASYNC_BUFFERS; i++) {
            if (inFlight_[i] == sequenceNo) {
                inFlight_[i] = 0;
                locked_--;
                break;
            }
        }
        assert(initial != locked_);
    }
}
#endif



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Setup %FBO instances to operate this layer (empty)
 *
 * As this layer does not require any %FBO to operate, this function is idle.
 */
void UploadLayer::setupFBOs() {
}


/**
 * @brief Upload input CPU buffer to texture(s)
 *
 * This functionn use <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml">glTexImage2D</a>
 * directly on the CPU buffers to (synchronously) update texture data to the GPU.
 */
void UploadLayer::syncUpload() {
    int rem = inputChannels_;
    int texoffs = 0;
    int width = width_ + 2 * inputPadding_;
    int height = height_ + 2 * inputPadding_;
    const uint8_t * srcptr = (const uint8_t *)input_->map<uint8_t>();
    if (!srcptr) {
        THROW_EXCEPTION_ARGS(FynException,"Cannot map source CPU buffer for (sync) texture upload");
    }
    while (rem > 0) {
        GLuint tex = outputTextures_.at(texoffs++);
        int chans = std::min(rem, LayerBase::PIXEL_PACKING);
        auto format = BufferSpec::formatByChannels(chans, dataType_);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, format.first, width, height, 0, format.second, dataType_, srcptr);
        srcptr += chans * width * height * bytesPerChan_;
        rem -= LayerBase::PIXEL_PACKING;    // we don't care about underflows
    }
    input_->unmap();
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Perform asynchronous upload operation
 *
 * @param sequenceNo Sequence number
 * @param callback Callback function into the engine when upload (copy portion) is done
 *
 * @retval true if asynchronous operation could commence
 * @retval false otherwise
 *
 * This function waits for an upload slot to become available, then fetches a ManagedPBO instance
 * to spawn the actual upload on (which runs in a different thread).
 */
bool UploadLayer::asyncUpload(uint64_t sequenceNo, const std::function<void(uint64_t)> & callback) {
    using namespace std::chrono_literals;
    //------------------------------------------------------------
    // Look for available upload slot, return failure if none
    // is available...
    //------------------------------------------------------------
    {
        std::unique_lock<std::mutex> asy(asyncLock_);
        int bufferidx = -1;
        if (locked_ < ASYNC_BUFFERS) {
            for (int i=0; i < ASYNC_BUFFERS; i++) {
                if (!inFlight_[i]) {
                    bufferidx = i;
                    break;
                }
            }
            if (bufferidx != -1) {
                inFlight_[bufferidx] = sequenceNo;
                locked_++;
            }
        } else {
            return false;
        }
        assert(bufferidx >= 0);
        //------------------------------------------------------------
        // Get PBO to buffer the CPU-side data for the upload and
        // schedule thread to handle the async upload...
        //------------------------------------------------------------
        const GLvoid * srcptr = input_->map<GLvoid>();
        if (!srcptr) {
            THROW_EXCEPTION_ARGS(FynException,"Cannot map source CPU buffer for (async) texture upload");
        }
        AsyncPool::GLThread thread = AsyncPool::getDerivedContextThread(context_);
        PBOPool *pool = context_.interface()->getWritePBOPool();
        assert(pool);
        ManagedPBO pbo = pool->getAvailablePBO(width_, height_, inputChannels_, bytesPerChan_);
        assert(!pbo.isPending());
        thread->setTask(std::bind(&UploadLayer::asyncUploadTask, this, pbo, srcptr, sequenceNo, input_, bufferidx, callback));
    }
    if (userCallback_) userCallback_(sequenceNo, input_, AsyncLayer::UPLOAD_COMMENCED);
    return true;
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Swap/set output textures to dependent layers based on sequence number
 *
 * @param sequence Sequence number of the sequence that is running with the next upload
 */
void UploadLayer::swapOutputTextures(uint64_t sequence) {
    std::unique_lock<std::mutex> asy(asyncLock_);
    const std::vector<GLuint> & textures = ((inFlight_[0] == sequence) ? outputTextures_ : shadowTextures_[0]);
    updateDependencies(textures);
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Task that performs (asynchronous) texture upload via a ManagedPBO instance
 *
 * @param pbo Reference to ManagedPBO that should be used to buffer the upload
 *
 * @param srcData Pointer to start of CPU buffer that contains the (correctly formatted) data to
 *                be uploaded
 *
 * @param sequence Sequence number which uniquely identifies/orders the operations/runs
 *
 * @param buffer Pointer to input CPUBuffer instance which will be unmapped by this thread once
 *               the upload was passed to the GL command queue
 *
 * @param texIdx Index of texture set to use as target
 *
 * @param callback Callback function to Engine that informs that the upload has been triggered
 *
 * @pre The \p buffer must be mapped into memory when calling this function and the result of that
 *      mapping must be supplied in \p srcData.
 *
 * @post The \p buffer will be unmapped from memory
 *
 * This function runs in a background thread and performs the actual texture upload by mapping a
 * %PBO into memory and copying the supplied \p srcData to it before invoking glTexImage2D()
 * to trigger the upload.
 *
 * @warning A word of advice regarding asynchronous texture uploads. In order to make sure not to
 *          overwrite %PBO buffer data \e before it was actually set up as texture (due to
 *          asynchronicity between the CPU and the GPU), this layer has to be "unlocked" before
 *          the next (asynchronous) texture upload can start. The unlocking must happen \e after
 *          \e all layers that consume the texture(s) written by this layer have consumed
 *          the data and written their own output. The only way to ensure that is by using appopriate
 *          fences.
 */
void UploadLayer::asyncUploadTask(opengl::ManagedPBO& pbo, const void *srcData, uint64_t sequence, CPUBuffer * buffer, int texIdx, const std::function<void(uint64_t)> & callback) {
    assert(srcData);
    assert(buffer);
    if (callback) {
        int width = width_ + 2 * inputPadding_;
        int height = height_ + 2 * inputPadding_;
        size_t totalsize = width * height * inputChannels_ * bytesPerChan_;
        // ------------------------------------------------
        // Copy data to PBO buffer...
        // ------------------------------------------------
        pbo->prepareForWrite(totalsize, true);
        void * pbobuffer = pbo->mapWriteBuffer(totalsize);
        assert(pbobuffer);
        memcpy(pbobuffer, srcData, totalsize);
        buffer->unmap();
        // ------------------------------------------------
        // The input buffer can be re-used now, if we have
        // a user callback function, notify it..
        // notify the engine...
        // ------------------------------------------------
        if (userCallback_) userCallback_(sequence, buffer, AsyncLayer::UPLOAD_DONE);
        // ------------------------------------------------
        // Upload PBO to textures...
        // ------------------------------------------------
        pbo->unmapWriteBuffer();
        int rem = inputChannels_;
        int texoffset = 0;
        uint32_t offset = 0;
        const std::vector<GLuint>& textures =  (texIdx == 0) ? outputTextures_ : shadowTextures_[texIdx-1];
        while (rem > 0) {
            int chans = std::min(rem, LayerBase::PIXEL_PACKING);
            auto format = BufferSpec::formatByChannels(chans, dataType_);
            GLuint tex = textures.at(texoffset);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, format.first, width, height, 0, format.second, dataType_, (const GLvoid *)(uintptr_t)offset);
            rem -= LayerBase::PIXEL_PACKING;        // we don't care about underflows
            offset += width * height * bytesPerChan_ * LayerBase::PIXEL_PACKING;
        }
        pbo->unbind(GL_PIXEL_UNPACK_BUFFER);
        // ------------------------------------------------
        // The texture generation is complete, notify the
        // engine that we may use it now...
        // ------------------------------------------------
        callback(sequence);
    } else {
        // TODO (mw) throw exception, as this should not really happen
        buffer->unmap();
    }
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Update input textures in dependent (receiving) layers
 *
 * @param textures New texture IDs to set in the dependent layers
 *
 * @pre #asyncLock_ held by current thread
 *
 * This function iterates over all receiving (dependent) layers of this upload layer and updates
 * the input texture IDs with the IDs from the current sequence number.
 */
void UploadLayer::updateDependencies(const std::vector<GLuint> &textures) const {
    for (int i=0; i < (int)dependencies_.size(); i++) {
        GPULayerBase * tgt = static_cast<GPULayerBase *>(dependencies_.at(i));
        int chanidx = dependencyOffsets_.at(i);
        for (int ti=0; ti < (int)textures.size(); ti++) {
            tgt->updateInputTexture(textures.at(ti), chanidx + ti);
        }
    }
}
#endif

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
