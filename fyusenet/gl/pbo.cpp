//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Pixelbuffer Object Wrapper
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <thread>

//-------------------------------------- Project  Headers ------------------------------------------

#include "pbo.h"
#include "pbopool.h"
#include "glexception.h"
#include "../common/miscdefs.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion::opengl {


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param width Width of the %PBO (in pixels)
 * @param height Height of the %PBO (in pixels)
 * @param channels Number of channels per pixel
 * @param bytesPerChan Bytes per channel
 * @param context Link to OpenGL context that the %PBO should associate to
 *
 * Generates an empty %PBO, no allocation is done here.
 */
PBO::PBO(int width, int height, int channels, int bytesPerChan, const fyusenet::GfxContextLink & context) :
    GLBuffer(GL_PIXEL_PACK_BUFFER, context), width_(width), height_(height), channels_(channels), bytesPerChannel_(bytesPerChan) {
    assert(handle_ != 0);
}


/**
 * @brief Copy CPU data into %PBO
 *
 * @param data Pointer to CPU buffer, may be \c nullptr if data is to be cleared/orphaned
 * @param dataSize Number of bytes to upload to the GPU
 * @param usage How to data is to be used (e.g. \c GL_STATIC_DRAW or \c GL_STREAM_DRAW )
 *
 * Uploads contents at \p data to the GPU. If a valid pointer was supplied, this function will
 * blockingly transfer the contents to the GPU. The driver may use optimization to keep the
 * CPU blocking part small and the data will only be fully available on the GPU a bit after this
 * function returns. In any case, the provided \p data can be deallocated / modified when this
 * function returns.
 *
 * @see https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming
 */
void PBO::setBufferData(const void *data, size_t dataSize, GLenum usage) {
    bind(GL_PIXEL_UNPACK_BUFFER);
    CLEAR_GFXERR_DEBUG
    if (bufferInit_) {
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, (GLsizeiptr)dataSize, data);
    } else {
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)dataSize, data, usage);
    }
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Cannot set buffer data for buffer %d target 0x%X (glerr=0x%X)",handle_,target_,err);
#endif
    unbind(GL_PIXEL_UNPACK_BUFFER);
    if (!bufferInit_) capacity_ = dataSize;
    bufferInit_ = true;
}


/**
 * @brief Change %PBO dimensions
 *
 * @param width New width
 * @param height New height
 * @param channels New number of channels
 * @param bytesPerChan New number of bytes per channel
 *
 * @post bufferInit_ is set to \c false in case the dimension has changed
 *
 * This changes the dimensions of the %PBO as stored internally, no modification to the underlying
 * GL object is done here.
 *
 * @note The number of \p channels supplied here might exceed the maximum number of channels per
 *       pixel (4), because a PBO is just treated as a buffer.
 */
void PBO::resize(int width, int height, int channels, int bytesPerChan) {
    width_ = width;
    height_ = height;
    bytesPerChannel_ = bytesPerChan;
    channels_ = channels;
    size_t size = width_ * height_ * channels_ * bytesPerChannel_;
    if (size > capacity_) {
        bufferInit_ = false;   // NOTE (mw) this allows growing the PBO
    }
}


/**
 * @brief Setup %PBO for persistent read operation
 *
 * @param dataSize Number of bytes to map into memory
 *
 * This function prepares the %PBO object to be used as memory-mapped I/O on a permanent/persistent
 * basis. That is, the %PBO stays mapped in memory until explicitly unmapped. To perform the
 * actual mapping (after this preparatory step) use the mapPersistentReadBuffer() function.
 *
 * @see flushForRead()
 * @see https://www.khronos.org/opengl/wiki/Buffer_Object#Persistent_mapping
 */
void PBO::prepareForPersistentRead(size_t dataSize) {
#if defined(FYUSENET_USE_EGL) || defined(__APPLE__) || defined(FYUSENET_USE_WEBGL)
    THROW_EXCEPTION_ARGS(GLNotImplException, "Persistent buffers are not implemented on this platform");
#else
    if (dataSize > capacity_) {
        if (!bound_) bind(GL_PIXEL_PACK_BUFFER);
#ifdef DEBUG
        glGetError();
#endif
        if (persistent_) glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        persistent_ = false;
        mapped_ = nullptr;
#if !defined(__APPLE__) && !defined(FYUSENET_USE_WEBGL)
        glBufferStorage(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)dataSize, nullptr, GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT);  // OpenGL 4.4+
#else
        glBufferData(GL_PIXEL_PACK_BUFFER, dataSize, nullptr, GL_STATIC_READ);
#endif
        bufferInit_ = true;
#ifdef DEBUG
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Cannot set buffer data for buffer %d target 0x%X (glerr=0x%X)",handle_,target_,err);
#endif
        unbind(GL_PIXEL_PACK_BUFFER);
        capacity_ = dataSize;
    }
#endif
}


/**
 * @brief Prepare a %PBO object for read operation (download from GPU)
 *
 * @param dataSize Number of bytes that should be allocated for this %PBO
 * @param leaveBound If set to \c true, the %PBO will remain bound after this call
 *
 * This function executes a preparatory step for a %PBO to be used for read operations. It basically
 * assigns an empty but dimensionalized buffer to the %PBO, which informs the GL subsystem of the
 * buffer size to be used for download.
 *
 * @see https://www.khronos.org/opengl/wiki/Buffer_Object
 */
void PBO::prepareForRead(size_t dataSize, bool leaveBound) {
    if ((dataSize > capacity_) || (!bufferInit_)) {
        bind(GL_PIXEL_PACK_BUFFER);
        CLEAR_GFXERR_DEBUG
        // NOTE (mw) do not use glBufferStorage() as it will fix the buffer size for the entire lifetime
        glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)dataSize, nullptr, GL_STREAM_READ);  // GL_STATIC_READ also works
        bufferInit_ = true;
#ifdef DEBUG
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Cannot set buffer data for buffer %d target 0x%X (glerr=0x%X, pbo=%p)",handle_, target_, err, this);
#endif
        if (!leaveBound) unbind(GL_PIXEL_PACK_BUFFER);
        capacity_ = dataSize;
    } else {
        if (leaveBound) bind(GL_PIXEL_PACK_BUFFER);
    }
}


/**
 * @brief Prepare a %PBO object for write operation (upload to GPU)
 *
 * @param dataSize Number of bytes that should be allocated for this %PBO
 * @param leaveBound If set to \c true, the %PBO will remain bound after this call
 *
 * This function executes a preparatory step for a %PBO to be used for write operations. It basically
 * assigns an empty but dimensionalized buffer to the %PBO, which informs the GL subsystem of the
 * buffer size to be used for download.
 *
 * @see https://www.khronos.org/opengl/wiki/Buffer_Object
 */
void PBO::prepareForWrite(size_t dataSize, bool leaveBound) {
    if ((dataSize > capacity_) || (!bufferInit_)) {
        bind(GL_PIXEL_UNPACK_BUFFER);
        CLEAR_GFXERR_DEBUG
        // NOTE (mw) do not use glBufferStorage() as it will fix the buffer size for the entire lifetime
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)dataSize, nullptr, GL_STREAM_DRAW);
        bufferInit_ = true;
#ifdef DEBUG
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Cannot set buffer data for buffer %d target 0x%X (glerr=0x%X, pbo=%p)",handle_, target_, err, this);
#endif
        if (!leaveBound) unbind(GL_PIXEL_UNPACK_BUFFER);
        capacity_ = dataSize;
    } else {
        if (leaveBound) bind(GL_PIXEL_UNPACK_BUFFER);
    }
}



/**
 * @brief Map %PBO as write target (upload to GPU)
 *
 * @param dataSize Size (can be size of range) of the buffer to map into CPU memory
 * @param offset Offset within the buffer to map into CPU memory
 * @param sync Use "synchronized" buffer access (i.e. do not set the \c GL_MAP_UNSYNCHRONIZED_BIT),
 *             which is the default. To use unsynchronized mapping, set this to \c false
 *
 * @return Pointer to memory address that maps (directly) to the buffer that backs the %PBO
 *
 * @post The %PBO will be bound
 *
 * This functions maps a %PBO as write target - and also dimensionalizes it when required - to
 * CPU memory.
 *
 * https://www.khronos.org/opengl/wiki/Buffer_Object
 */
void * PBO::mapWriteBuffer(size_t dataSize, size_t offset, bool sync) {
#ifdef FYUSENET_USE_WEBGL
    THROW_EXCEPTION_ARGS(GLNotImplException,"WebGL does not support mapping buffers");
#else
    assert(bufferInit_);
    if (!bound_) bind(GL_PIXEL_UNPACK_BUFFER);
#ifdef DEBUG
    glGetError();
#endif
    void * ptr = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, (GLintptr)offset, (GLsizeiptr)dataSize, GL_MAP_WRITE_BIT|GL_MAP_INVALIDATE_BUFFER_BIT);
#ifdef DEBUG
    GLenum err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
    return ptr;
#endif
}

/**
 * @brief Map read-only memory of %PBO to CPU memory persistently
 *
 * @param dataSize Number of bytes to map into memory
 *
 * @return Pointer to mapped memory, stays valid (also across threads) until it is unmapped
 *
 * This function maps the %PBO into the processes virtual memory on a permanent/persistent basis.
 * That is, the %PBO stays mapped in memory until explicitly unmapped. Note that the API user is
 * responsible for ensuring that a mapped buffer is synchronized properly with shader operations
 * by issueing appropriate memory barriers and sync (fence) operations.
 *
 * @see flushForRead()
 * @see https://www.khronos.org/opengl/wiki/Buffer_Object#Persistent_mapping
 */
void * PBO::mapPersistentReadBuffer(size_t dataSize) {
#if defined(FYUSENET_USE_EGL) || defined(__APPLE__) || defined(FYUSENET_USE_WEBGL)
    THROW_EXCEPTION_ARGS(GLNotImplException, "Persistent buffers are not implemented on this platform");
#else
    if (!persistent_) {
        persistent_ = true;
        mapped_ = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)dataSize, GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT);
    }
    assert(mapped_);
    return mapped_;
#endif
}


/**
 * @brief Convenience function that maps a ready-only %PBO based on its initial size
 *
 * @return Pointer to mapped memory, stays valid (also across threads) until it is unmapped
 */
void * PBO::mapPersistentReadBuffer() {
    return mapPersistentReadBuffer(capacity_);
}


/**
 * @brief Map read-only memory of %PBO to CPU memory
 *
 * @param dataSize Size (can be size of range) of the buffer to map into CPU memory
 * @param offset Offset within the buffer to map into CPU memory
 *
 * @return Pointer to mapped memory
 */
void * PBO::mapReadBuffer(size_t dataSize, size_t offset) {
#ifdef FYUSENET_USE_WEBGL
    THROW_EXCEPTION_ARGS(GLNotImplException,"WebGL does not support mapping buffers");
#else
    if (!bound_) bind(GL_PIXEL_PACK_BUFFER);
    assert(bufferInit_);
    assert(dataSize <= capacity_);
    CLEAR_GFXERR_DEBUG
    void * ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, (GLintptr)offset, (GLsizeiptr)dataSize, GL_MAP_READ_BIT);
#ifdef DEBUG
    GLenum err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
    return ptr;
#endif
}


/**
 * @brief Convenience function that maps a ready-only %PBO based on its initial size
 *
 * @return Pointer to mapped memory
 */
void * PBO::mapReadBuffer() {
    assert(capacity_ > 0);
    return mapReadBuffer(capacity_, 0);
}


/**
 * @brief Unmap memory from %PBO
 *
 * @pre The %PBO must be bound
 * @post %PBO is still bound
 */
void PBO::unmapReadBuffer() {
#ifdef FYUSENET_USE_WEBGL
    THROW_EXCEPTION_ARGS(GLException,"WebGL does not support mapping buffers");
#else
    if (!bound_) THROW_EXCEPTION_ARGS(GLException,"PBO not bound");    
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
#endif
}

/**
 * @brief Unmap memory from %PBO
 *
 * @pre The %PBO must be bound
 * @post %PBO is still bound
 */
void PBO::unmapWriteBuffer() {
#ifdef FYUSENET_USE_WEBGL
    THROW_EXCEPTION_ARGS(GLException,"WebGL does not support mapping buffers");
#else
    if (!bound_) THROW_EXCEPTION_ARGS(GLException,"PBO not bound");
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
#endif
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // fyusion::opengl namespace



// vim: set expandtab ts=4 sw=4:
