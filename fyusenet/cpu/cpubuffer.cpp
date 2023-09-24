//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Buffer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "cpubuffer.h"
#include "../common/miscdefs.h"
#include "../gl/pbo.h"
#include "../gpu/deep/deeptiler.h"
#include "../base/layerbase.h"

namespace fyusion::fyusenet::cpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param shape Shape descriptor to construct a buffer for
 */
CPUBuffer::CPUBuffer(const BufferShape& shape) : shape_(shape) {
    if (shape.bytes() > 0) {
        memory_ = malloc(shape.bytes());
        if (!memory_) {
            FNLOGE("Cannot allocate CPU buffer");
            throw std::bad_alloc();
        }
    }
}



/**
 * @brief Destructor
 */
CPUBuffer::~CPUBuffer() {
    FNET_DEL_AND_CLEAR(tiler_);
    mapped_.lock();
    if (memory_) free(memory_);
    memory_ = nullptr;
    mapped_.unlock();
}


/**
 * @brief Retrieve buffer capacity in bytes
 *
 * @return Number of bytes that the buffer can store in total
 */
size_t CPUBuffer::bytes() const {
    return shape_.bytes();
}


/**
 * @brief Perform a deep copy of the buffer to a (new/other) buffer
 *
 * @param tgt Optional target buffer to perform copy to, when \c nullptr is supplied, a new buffer
 *            will be created
 *
 * @return Target buffer that holds a deep-copy of the data of the current buffer.
 *
 * Use this function is a deep-copy of a buffer is required. For buffers that are based on PBOs,
 * this function copies the data from the memory-mapped area to "normal" CPU memory.
 */
CPUBuffer * CPUBuffer::copyTo(CPUBuffer * tgt) const {
    if (!tgt) tgt = shape_.createCPUBuffer();
    else {
        if (shape_ != tgt->shape_) THROW_EXCEPTION_ARGS(FynException,"Cannot copy buffer to incompatible target buffer");
    }
    const auto * srcdata = map<uint8_t>();
    auto * tgtdata = tgt->map<uint8_t>();
    assert(srcdata);
    assert(tgtdata);
    memcpy(tgtdata, srcdata, shape_.bytes());
    unmap();
    tgt->unmap();
    return tgt;
}


/**
 * @brief Convert buffer instance to channel-wise data storage order
 *
 * @param tgt Pointer to target buffer, or \c nullptr in which case the target buffer will be
 *            created
 *
 * @return Target buffer with channel-wise data storage order, or \c nullptr if no conversion was
 *         possible
 *
 * @warning This implementation is incomplete, in its current state it can only copy channel-wise
 *          CPU buffers
 */
CPUBuffer * CPUBuffer::toChannelWise(CPUBuffer *tgt) const {
    if (!tgt) {
        tgt = shape_.createCPUBuffer(BufferShape::order::CHANNELWISE);
    }
    else {
        auto shape = shape_.asOrder(BufferShape::order::CHANNELWISE);
        if (!tgt->shape_.sameSize(shape) || !tgt->shape_.sameType(shape) ||
            tgt->shape_.dataOrder() != BufferShape::order::CHANNELWISE) {
            THROW_EXCEPTION_ARGS(FynException,"Mismatching shapes");
        }
    }
    if ((shape_.dataOrder_ == BufferShape::order::CHANNELWISE) ||
        ((shape_.width_ == 1) && (shape_.height_ == 1))) {
        auto * srcdata = map<uint8_t>();
        auto * tgtdata = tgt->map<uint8_t>();
        assert(srcdata);
        assert(tgtdata);
        memcpy(tgtdata, srcdata, bytes());
        unmap();
        tgt->unmap();
        return tgt;
    } else if (shape_.dataOrder_ == BufferShape::order::GPU_DEEP) {
        const float * srcdata = map<float>();
        float * tgtdata = tgt->map<float>();
        deepToChannelWise<float>(srcdata, tgtdata);
        unmap();
        tgt->unmap();
        return tgt;
    } else if (shape_.dataOrder_ == BufferShape::order::GPU_SHALLOW) {
        const float * srcdata = map<float>();
        float * tgtdata = tgt->map<float>();
        shallowToChannelWise<float>(srcdata, tgtdata);
        unmap();
        tgt->unmap();
        return tgt;
    }
    return nullptr;
}


/**
 * @brief Convert current buffer instance to GPU shallow-tensor data storage order
 *
 * @param tgt Pointer to target buffer, or \c nullptr in which case the target buffer will be
 *            created
 *
 * @return Target buffer with GPU shallow data representation storage order
 *
 *
 * @warning This implementation is incomplete, in its current state it can only copy GPU
 *          shallow-tensor ordered CPU buffers
 */
CPUBuffer * CPUBuffer::toGPUShallow(CPUBuffer *tgt) const {
    if (shape_.dataOrder_ != BufferShape::order::GPU_SHALLOW) {
        // TODO (mw) implement missing conversions
        THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
    }
    if (!tgt) tgt = shape_.createCPUBuffer(BufferShape::order::GPU_SHALLOW);
    else {
        if (!tgt->shape().sameOrder(shape_)) {
            THROW_EXCEPTION_ARGS(FynException,"Data order of target buffer is not compatible");
        }
    }
    const auto * srcdata = map<uint8_t>();
    auto * tgtdata = tgt->map<uint8_t>();
    assert(srcdata);
    assert(tgtdata);
    memcpy(tgtdata, srcdata, bytes());
    unmap();
    tgt->unmap();
    return tgt;
}


/**
 * @brief Convert current buffer instance to GPU deep-tensor data storage order
 *
 * @param tgt Pointer to target buffer, or \c nullptr in which case the target buffer will be
 *            created
 *
 * @return Target buffer with GPU deep-tensor data storage order
 *
 * @warning This implementation is incomplete, in its current state it can only copy GPU deep-tensor
 *          ordered CPU buffers
 */
CPUBuffer * CPUBuffer::toGPUDeep(CPUBuffer *tgt) const {
    if (!tgt) tgt = shape_.createCPUBuffer(BufferShape::order::GPU_DEEP);
    else {
        // TODO (mw) complete the implementation here
        THROW_EXCEPTION_ARGS(FynException, "Incomplete implementation");
    }
    if (shape_.dataOrder_ == BufferShape::order::GPU_DEEP) {
        const auto * srcdata = map<uint8_t>();
        auto * tgtdata = tgt->map<uint8_t>();
        assert(srcdata);
        assert(tgtdata);
        memcpy(tgtdata, srcdata, bytes());
        unmap();
        tgt->unmap();
        return tgt;
    } else {
        THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
    }
}



/**
 * @brief Dump the contents of this CPUBuffer to a file
 *
 * @param fileName Name of file to write the data into
 *
 * This is a debug convenience function that writes out a CPUBuffer to a specified file. Prior
 * to writing, the tensor data is reformatted such that it is arranged channel-wise as simple 3D
 * array.
 *
 * @see LayerBase::writeResult
 */
template<typename T>
void CPUBuffer::write(const char *fileName) const {
#ifdef DEBUG
#ifndef FYUSENET_USE_WEBGL
    FILE *out = fopen(fileName,"wb");
    if (!out) THROW_EXCEPTION_ARGS(FynException,"Cannot open file %s for writing", fileName);
#endif
    switch (shape_.dataOrder_) {
        case BufferShape::order::CHANNELWISE:
#ifndef FYUSENET_USE_WEBGL
            fwrite(map<uint8_t>(),1,shape_.bytes(),out);
#else
            EM_ASM({window.download($0, $1, $2);}, map<uint8_t>(), shape_.bytes(), fileName);
#endif
            unmap();
            break;
        case BufferShape::order::GPU_SHALLOW: {
            T * tmp = (T *)malloc(shape_.bytes(BufferShape::order::CHANNELWISE));
            if (!tmp) {
                FNLOGE("Cannot allocate tmp buffer");
                throw std::bad_alloc();
            }
            const T *src = map<T>();
            shallowToChannelWise<T>(src, tmp);
            unmap();
#ifndef FYUSENET_USE_WEBGL
            fwrite(tmp, 1, shape_.bytes(BufferShape::order::CHANNELWISE), out);
#else
            EM_ASM({window.download($0, $1, $2);}, tmp, shape_.bytes(BufferShape::order::CHANNELWISE), fileName);
#endif
            free(tmp);
        }
        break;
        case BufferShape::order::GPU_DEEP: {
            T * tmp = (T *)malloc(shape_.bytes(BufferShape::order::CHANNELWISE));
            if (!tmp) {
                FNLOGE("Cannot allocate tmp buffer");
                throw std::bad_alloc();
            }
            const T *src = map<T>();
            deepToChannelWise<T>(src, tmp);
            unmap();
#ifndef FYUSENET_USE_WEBGL
            fwrite(tmp, 1, shape_.bytes(BufferShape::order::CHANNELWISE), out);
#else
            EM_ASM({window.download($0, $1, $2);}, tmp, shape_.bytes(BufferShape::order::CHANNELWISE), fileName);
#endif
            free(tmp);
        }
        break;
        case BufferShape::order::GPU_SEQUENCE: {
            // NOTE (mw) for sequences we do not convert to channel-wise, but rather write out as is
            const T * src = map<T>();
#ifndef FYUSENET_USE_WEBGL
            fwrite(src, 1, shape_.bytes(), out);
#else
            EM_ASM({window.download($0, $1, $2);}, src, shape_.bytes(), fileName);
#endif
            unmap();
        }
        break;
        default:
            THROW_EXCEPTION_ARGS(FynException, "Unsupported data order");
    }
#ifndef FYUSENET_USE_WEBGL
    fclose(out);
#endif
#endif
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Read data from PBO into CPUBuffer
 *
 * @param pbo Source PBO to read data from
 *
 * @param type Data type of the CPU buffer
 *
 * @param sequenceNo Sequence number to assign to this buffer (which should be the sequence number
 *                   of the content currently in the %PBO)
 *
 * @param bytes (Optional) number of bytes to read from the PBO, if supplied with zero, it will
 *              read the full PBO contents
 *
 * @retval true if read operation was successful
 * @retval false otherwise
 *
 * This function reads the content of the supplied \p pbo into this buffer instance.
 *
 * @warning This function currently only supports \c FLOAT32 data types
 */
bool CPUBuffer::readFromPBO(opengl::PBO * pbo, BufferShape::type type, uint64_t sequenceNo, size_t bytes) {
    if (!memory_) return false;
    CLEAR_GFXERR_DEBUG
    pbo->bind(GL_PIXEL_PACK_BUFFER);
    auto * tgt = map<uint8_t>();
    if (!tgt) THROW_EXCEPTION_ARGS(FynException,"Oops, trying to copy to an already mapped buffer");
    size_t cap = this->bytes();
    void * src = pbo->mapReadBuffer();
    if (!src) {
        unmap();
        THROW_EXCEPTION_ARGS(FynException,"Cannot read data from PBO");
    }
#ifdef DEBUG
    GLenum err = glGetError();
    if (err) {
        FNLOGE("Cannot map PBO buffer, err=0x%x src=%p",err,src);
        pbo->unmapReadBuffer();
        pbo->unbind(GL_PIXEL_PACK_BUFFER);
        unmap();
        return false;
    }
#endif
    size_t sz = (bytes == 0) ? pbo->capacity() : bytes;
    if (sz > cap) {
        pbo->unmapReadBuffer();
        pbo->unbind(GL_PIXEL_PACK_BUFFER);
        unmap();
        THROW_EXCEPTION_ARGS(FynException,"Refusing to read from PBO as this would exceed buffer size");
    }
    memcpy(tgt, src, sz);
    unmap();
    pbo->unmapReadBuffer();
    pbo->unbind();
    sequenceNo_ = sequenceNo;
    return true;
}



/**
 * @brief Reformat deep-tensor GPU data to channel-wise format
 *
 * @param src Pointer to source (raw) buffer
 * @param tgt Pointer to target (raw) buffer
 *
 * This function reformats the supplied \p src buffer from GPU deep-tensor format into a plain
 * channel-wise format, that represents the tensor as simple 3D array.
 */
template<typename T>
void CPUBuffer::deepToChannelWise(const T *src, T *tgt) const {
    // NOTE (mw) unoptimized code
    assert(shape_.padding_ <= 1);
    if (!tiler_) {
        tiler_ = new gpu::deep::DeepTiler(LayerType::DOWNLOAD,
                                          shape_.tileWidth_, shape_.tileHeight_, shape_.channels_, shape_.channels_,
                                          1.0f, 1.0f, 0, shape_.padding_, 1, 1, 1, 1);
    }
    int pad = shape_.padding_;
    int lwidth = shape_.tileWidth_ + 2 * pad;
    int lheight = shape_.tileHeight_ + 2 * pad;
    int channeloffset = 0;
    int srcstride = shape_.width_ * LayerBase::PIXEL_PACKING;
    for (int ty=0; ty < tiler_->numOutputTiles(gpu::deep::DeepTiler::VERTICAL); ty++) {
        for (int tx=0; tx < tiler_->numOutputTiles(gpu::deep::DeepTiler::HORIZONTAL); tx++) {
            int rem = ((shape_.channels_ - channeloffset) > LayerBase::PIXEL_PACKING) ? LayerBase::PIXEL_PACKING : shape_.channels_ - channeloffset;
            const T * in = src +  (ty * (shape_.tileHeight_ + pad)) * srcstride + (tx * shape_.tileWidth_ + pad) * LayerBase::PIXEL_PACKING;
            for (int l=0; l < rem; l++) {
                T * outptr = tgt + channeloffset * (lwidth * lheight);
                for (int y=0; y < lheight; y++) {
                    for (int x=0; x < lwidth; x++) {
                        outptr[x+y*lwidth] = in[(y*srcstride+x)*LayerBase::PIXEL_PACKING+l];
                    }
                }
                channeloffset++;
            }
        }
    }
}



/**
 * @brief Reformat shallow-tensor GPU data to channel-wise format
 *
 * @param src Pointer to source (raw) buffer
 * @param tgt Pointer to target (raw) buffer
 * @param channelOffset First channel in the source to start reformatting
 *
 * This function reformats the supplied \p src buffer from GPU shallow-tensor format into a plain
 * channel-wise format, that represents the tensor as simple 3D array.
 */
template<typename T>
void CPUBuffer::shallowToChannelWise(const T *src, T *tgt, int channelOffset) const {
    // NOTE (mw) unoptimized code
    int lwidth = shape_.width_ + 2*shape_.padding_;
    int lheight = shape_.height_ + 2*shape_.padding_;
    int rem = shape_.channels_ - channelOffset;
    if (rem > LayerBase::PIXEL_PACKING) rem = LayerBase::PIXEL_PACKING;
    for (int l = 0 ; l < rem; l++) {
        for (int y=0; y < lheight; y++) {
            for (int x=0; x < lwidth; x++) {
                tgt[x+y*lwidth]=src[l+channelOffset+y*lwidth+x];
            }
        }
    }
}



/*##################################################################################################
#                 E X P L I C I T   T E M P L A T E   I N S T A N T I A T I O N S                  #
##################################################################################################*/

template void CPUBuffer::write<float>(const char *fileName) const;
#ifndef FYUSENET_CPU_FLOAT_ONLY
extern template void CPUBuffer::write<uint32_t>(const char *fileName) const;
template void CPUBuffer::write<uint16_t>(const char *fileName) const;
template void CPUBuffer::write<uint8_t>(const char *fileName) const;
template void CPUBuffer::write<int32_t>(const char *fileName) const;
template void CPUBuffer::write<int16_t>(const char *fileName) const;
template void CPUBuffer::write<int8_t>(const char *fileName) const;
#endif

template void CPUBuffer::deepToChannelWise<float>(const float *src, float *tgt) const;
#ifndef FYUSENET_CPU_FLOAT_ONLY
template void CPUBuffer::deepToChannelWise<uint32_t>(const uint32_t *src,uint32_t *tgt) const;
template void CPUBuffer::deepToChannelWise<uint16_t>(const uint16_t *src,uint16_t *tgt) const;
template void CPUBuffer::deepToChannelWise<uint8_t>(const uint8_t *src,uint8_t *tgt) const;
template void CPUBuffer::deepToChannelWise<int32_t>(const int32_t *src,int32_t *tgt) const;
template void CPUBuffer::deepToChannelWise<int16_t>(const int16_t *src,int16_t *tgt) const;
template void CPUBuffer::deepToChannelWise<int8_t>(const int8_t *src,int8_t *tgt) const;
#endif

template void CPUBuffer::shallowToChannelWise<float>(const float * src, float *tgt, int channelOffset=0) const;
#ifndef FYUSENET_CPU_FLOAT_ONLY
template void CPUBuffer::shallowToChannelWise<uint32_t>(const uint32_t * src,uint32_t *tgt, int channelOffset=0) const;
template void CPUBuffer::shallowToChannelWise<uint16_t>(const uint16_t * src,uint16_t *tgt, int channelOffset=0) const;
template void CPUBuffer::shallowToChannelWise<uint8_t>(const uint8_t * src,uint8_t *tgt, int channelOffset=0) const;
template void CPUBuffer::shallowToChannelWise<int32_t>(const int32_t * src,int32_t *tgt, int channelOffset=0) const;
template void CPUBuffer::shallowToChannelWise<int16_t>(const int16_t * src,int16_t *tgt, int channelOffset=0) const;
template void CPUBuffer::shallowToChannelWise<int8_t>(const int8_t * src,int8_t *tgt, int channelOffset=0) const;
#endif


} // fyusion::fyusenet::cpu namespace

// vim: set expandtab ts=4 sw=4:
