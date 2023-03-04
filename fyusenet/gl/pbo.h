//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Pixelbuffer Object Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <atomic>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "glbuffer.h"
#include "../gpu/gfxcontextlink.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace opengl {

/**
 * @brief Wrapper class around OpenGL pixel buffer objects (PBOs)
 *
 * This class wraps an OpenGL pixel buffer object into a (more) convenient class representation.
 * Pixel buffer objects can be viewed as (asynchronous) gateways to/from GPU memory and if used
 * correctly, are able to hide data latencies in texture upload and texture download for
 * maximum throughput. In almost all cases, PBOs should be used asynchronously, such that
 * data transfer operations can be done by a background or data-transfer thread in order to prevent
 * the main/render thread from waiting.
 *
 * A %PBO object is usually either used in conjunction with an FBO (for GPU -> CPU transfer) or with
 * a texture object (for CPU -> GPU  transfer).
 *
 * @todo Split this class into a %PBO for reading and a %PBO for writing
 *
 * @see https://www.khronos.org/opengl/wiki/Pixel_Buffer_Object
 */
class PBO : public GLBuffer {
 public:
    enum accesstype {
        READ,
        WRITE
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    PBO(int width, int height, int channels, int bytesPerChannel, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setBufferData(const void *data, size_t dataSize, GLenum usage);
    void * mapPersistentReadBuffer(size_t dataSize);
    void * mapPersistentReadBuffer();
    void * mapReadBuffer(size_t dataSize, size_t offset=0);
    void * mapReadBuffer();
    void * mapWriteBuffer(size_t dataSize, size_t offset=0, bool sync=true);
    void unmapReadBuffer();
    void unmapWriteBuffer();
    void prepareForPersistentRead(size_t dataSize);
    void prepareForRead(size_t dataSize, bool leaveBound=false);
    void prepareForWrite(size_t dataSize, bool leaveBound=false);
    void resize(int width, int height, int channels, int bytesPerChan);
    void writeToMemory(void *data);


    bool matches(int width, int height, int channels, int bytesPerChannel) const {
        // NOTE (mw) we are way too strict here, a PBO is just a buffer and a too large PBO will
        // do the job just fine, but we leave it at this for now
        return (width_ == width) && (height_ == height) && (channels_ == channels) && (bytesPerChannel == bytesPerChannel_);
    }

    /**
     * @brief Get width of %PBO (in pixels)
     *
     * @return Width of %PBO
     */
    int width() const {
        return width_;
    }

    /**
     * @brief Get height of %PBO (in pixels)
     *
     * @return Height of %PBO
     */
    int height() const {
        return height_;
    }

    /**
     * @brief Get number of channels for %PBO
     *
     * @return Number of channels per pixel
     */
    int channels() const {
        return channels_;
    }

    /**
     * @brief Retrieve capacity of this object in bytes
     *
     * @return Number of bytes mappable in the buffer
     *
     * @pre prepareForRead() or prepareForWrite() has been called
     */
    size_t capacity() const {
        return capacity_;
    }


    /**
     * @brief Flush data from shader into client memory for persistent PBOs
     */
    void flushForRead() {
#if !defined(__APPLE__) && !defined(FYUSENET_USE_EGL) && !defined(FYUSENET_USE_WEBGL)
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
#else
        THROW_EXCEPTION_ARGS(GLNotImplException,"Memory barriers not implemented on this platform");
#endif
    }

 protected:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;                         //!< Width in pixels
    int height_ = 0;                        //!< Height in pixels
    uint8_t channels_ = 0;                  //!< Channels per pixel (in [1..4])
    uint8_t bytesPerChannel_ = 0;           //!< Bytes per channel
    size_t capacity_ = 0;                   //!< buffer size "allocated"
    void * mapped_ = nullptr;               //!< For persistent PBOs: host memory address
    bool bufferInit_ = false;               //!< Indicator if buffer was initialized (either for reading or writing)
    bool persistent_ = false;               //!< Indicator if buffer is a persistent buffer
};


} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
