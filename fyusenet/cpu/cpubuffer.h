//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Buffer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <memory>
#include <atomic>
#include <functional>
#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../common/fynexception.h"
#include "cpubuffershape.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {

namespace opengl {
    class PBO;
}

namespace fyusenet {

namespace gpu {
    class DownloadLayer;
    namespace deep {
        class DeepDownloadLayer;
    }
}

namespace cpu {


/**
 * @brief General CPU buffer class to wrap tensor data accessible by CPU
 *
 * This class is always used in conjunction with a CPUBufferShape object, which assigns structural
 * information to the buffer. As FyuseNet is not aimed at running too much things on the CPU, the
 * CPU buffers main functionality is to provide a means of access to texture data, either by
 * first downloading into main memory directly in a synchronized fashion, or by downloading into
 * a PBO. This type of access requires a few constraints which might seem a bit overboard, but
 * for now we go with that.
 *
 * In order to access the content of a CPUBuffer, a call to map() will provide a (raw) pointer to
 * the data stored in the buffer. This call \b must be matched with a call to unmap() after the
 * access has done. Failure to do so will result in the buffer returning a \c nullptr on the
 * next call to map() that does not involve waiting. The reason for using the map/unmap construct
 * is to serialize access to a CPU buffer and also to be able to realize future (internal)
 * expansions of the CPUBuffer with regards to directly wrapping a PBO or another GPU-based
 * memory-mapping to avoid data copy.
 *
 * The current implementation interfaces with a PBO by copying the data in order to release the
 * source PBO as soon as possible, but this may change in the future.
 */
class CPUBuffer {
    friend class CPUBufferShape;
    friend class gpu::DownloadLayer;
    friend class gpu::deep::DeepDownloadLayer;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    CPUBuffer(const CPUBufferShape& shape);
    CPUBuffer(const CPUBuffer&) = delete;
    CPUBuffer& operator=(const CPUBuffer&) = delete;
    ~CPUBuffer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    size_t bytes() const;
    CPUBuffer * copyTo(CPUBuffer * tgt = nullptr) const;
    CPUBuffer * toChannelWise(CPUBuffer *tgt = nullptr) const;
    CPUBuffer * toGPUShallow(CPUBuffer *tgt = nullptr) const;
    CPUBuffer * toGPUDeep(CPUBuffer *tgt = nullptr) const;

    /**
     * @brief Associate CPU buffer content with a sequence ID
     *
     * @param sequence Sequence ID to store in this buffer
     *
     * @see NeuralNetwork::forward
     */
    void associateTo(uint64_t sequence) {
        sequenceNo_ = sequence;
    }


    /**
     * @brief Fill CPU buffer with single value
     *
     * @param value Value to fill buffer with
     */
    template<typename T>
    void fill(T value) {
        if (!memory_) THROW_EXCEPTION_ARGS(FynException,"Cannot fill null buffer");
        // TODO (mw) handle PBOs in case direct PBO mapping is implemented at some point
        for (int i=0; i < bytes()/sizeof(T); i++) ((T *)memory_)[i] = value;
    }

    /**
     * @brief Retrieve shape for this buffer
     *
     * @return CPUBufferShape instance that stores the buffer shape
     */
    const CPUBufferShape & shape() const {
        return shape_;
    }

    /**
     * @brief Retrieve sequence ID associated with this buffer
     *
     * @return Sequence ID
     *
     * @see NeuralNetwork::forward
     */
    uint64_t sequence() const {
        return sequenceNo_;
    }


    /**
     * @brief Map data stored in this object to memory and retrieve (read only) pointer
     *
     * @return Pointer to data in this object for reading purposes or \c nullptr if mapping could
     *         not be done
     *
     * See class description for details on mapping/unmapping.
     */
    template<typename T>
    const T * map(bool wait=false) const {
        bool gotmilk = mapped_.try_lock();
        if (!gotmilk) {
             if (!wait) return nullptr;
             mapped_.lock();
        }
        const T * ptr = (const T *)memory_;
        return ptr;
    }


    /**
     * @brief Map data stored in this object to memory and retrieve (writable) pointer
     *
     * @return Pointer to data in this object for reading purposes or \c nullptr if mapping could
     *         not be done
     *
     * See class description for details on mapping/unmapping.
     */
    template<typename T>
    T * map(bool wait=false) {
        bool gotmilk = mapped_.try_lock();
        if (!gotmilk) {
            if (!wait) return nullptr;
            mapped_.lock();
        }
        T * ptr = (T *)memory_;
        return ptr;
    }


    /**
     * @brief Unmap CPU buffer from memory
     *
     * @warning For some CPU buffers, accessing a previously obtained pointer via the map()
     *          function may still work after unmapping, however there is no guarantee that it will
     *          with all buffers. For that reason, discard all raw pointers obtained from map()
     *          in your implementation when unmapping the buffer to prevent illegal memory access.
     */
    void unmap() const {
        mapped_.unlock();
    }


    /**
     * @brief Execute function that uses this buffer for reading and writing
     *
     * @param func Function to execute
     *
     * This function maps the buffer into memory for reading nand writing, then executes the
     * supplied function, providing a pointer to the buffer as parameter. After execution of the
     * supplied function, the buffer will be unmapped again.
     *
     * @note Using this function is not an atomic operation on the buffer and its contents
     */
    template<typename T>
    void with(std::function<void(T *)>& func) {
        func(map<T>());
        unmap();
    }

    /**
     * @brief Execute function that uses this buffer for reading
     *
     * @param func Function to execute
     *
     * This function maps the buffer into memory for reading, then executes the supplied function,
     * providing a pointer to the buffer as parameter. After execution of the supplied function,
     * the buffer will be unmapped again.
     *
     * @note Using this function is not an atomic operation on the buffer and its contents
     */
    template<typename T>
    void with(std::function<void(const T *)> func) const {
        func(map<T>());
        unmap();
    }

    template<typename T>
    void write(const char *fileName) const;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    bool readFromPBO(opengl::PBO *pbo, CPUBufferShape::type type, uint64_t sequenceNo);
    static GLuint typeToGLType(CPUBufferShape::type type);
    template<typename T>
    void shallowToChannelWise(const T *src, T *tgt, int channelOffset=0) const;
    template<typename T>
    void deepToChannelWise(const T *src, T *tgt) const;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    CPUBufferShape shape_;                    //!< Shape for this buffer
    void * memory_ = nullptr;                 //!< Pointer to buffer memory
    uint64_t sequenceNo_ = 0;                 //!< Sequence number that the contents of this buffer are associated to (optional)
    /**
     * Lock/Indicator if buffer is mapped
     */    
    mutable std::mutex mapped_;               // TODO (mw) think about using R/W locks here instead
    /**
     * Deep-tensor tile computation used in conversion code
     */
    mutable gpu::deep::DeepTiler *tiler_ = nullptr;
};


/**
 * @brief Represention for typed CPU buffers
 *
 * This is a convenience mechanism that explicitly assigned data-types to CPU buffers, which
 * usually do not convey their internal data type in their signature.
 */
template<typename T>
class TypedCPUBuffer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    TypedCPUBuffer(CPUBuffer * wrap);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    T * map(bool wait=false) {
        return buffer_->map<T>();
    }

    const T * map(bool wait=false) const {
        return buffer_->map<T>(wait);
    }

    void unmap() const {
        buffer_->unmap();
    }

protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    mutable CPUBuffer * buffer_ = nullptr;
};



extern template void CPUBuffer::write<float>(const char *fileName) const;
#ifndef FYUSENET_CPU_FLOAT_ONLY
extern template void CPUBuffer::write<uint32_t>(const char *fileName) const;
extern template void CPUBuffer::write<uint16_t>(const char *fileName) const;
extern template void CPUBuffer::write<uint8_t>(const char *fileName) const;
extern template void CPUBuffer::write<int32_t>(const char *fileName) const;
extern template void CPUBuffer::write<int16_t>(const char *fileName) const;
extern template void CPUBuffer::write<int8_t>(const char *fileName) const;
#endif


extern template void CPUBuffer::deepToChannelWise<float>(const float *src, float *tgt) const;
#ifndef FYUSENET_CPU_FLOAT_ONLY
extern template void CPUBuffer::deepToChannelWise<uint32_t>(const uint32_t *src, uint32_t *tgt) const;
extern template void CPUBuffer::deepToChannelWise<uint16_t>(const uint16_t *src, uint16_t *tgt) const;
extern template void CPUBuffer::deepToChannelWise<uint8_t>(const uint8_t *src, uint8_t *tgt) const;
extern template void CPUBuffer::deepToChannelWise<int32_t>(const int32_t *src, int32_t *tgt) const;
extern template void CPUBuffer::deepToChannelWise<int16_t>(const int16_t *src, int16_t *tgt) const;
extern template void CPUBuffer::deepToChannelWise<int8_t>(const int8_t *src, int8_t *tgt) const;
#endif

extern template void CPUBuffer::shallowToChannelWise<float>(const float * src, float *tgt, int channelOffset=0) const;
#ifndef FYUSENET_CPU_FLOAT_ONLY
extern template void CPUBuffer::shallowToChannelWise<uint32_t>(const uint32_t * src, uint32_t *tgt, int channelOffset=0) const;
extern template void CPUBuffer::shallowToChannelWise<uint16_t>(const uint16_t * src, uint16_t *tgt, int channelOffset=0) const;
extern template void CPUBuffer::shallowToChannelWise<uint8_t>(const uint8_t * src, uint8_t *tgt, int channelOffset=0) const;
extern template void CPUBuffer::shallowToChannelWise<int32_t>(const int32_t * src, int32_t *tgt, int channelOffset=0) const;
extern template void CPUBuffer::shallowToChannelWise<int16_t>(const int16_t * src, int16_t *tgt, int channelOffset=0) const;
extern template void CPUBuffer::shallowToChannelWise<int8_t>(const int8_t * src, int8_t *tgt, int channelOffset=0) const;
#endif


} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:

