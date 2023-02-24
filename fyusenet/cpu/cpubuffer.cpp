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
#include <limits>
#include <inttypes.h>


//-------------------------------------- Project  Headers ------------------------------------------

#include "cpubuffer.h"
#include "../gl/gl_sys.h"
#include "../gl/pbo.h"
#include "../gpu/deep/deeptiler.h"
#include "../base/layerbase.h"
#include "../common/logging.h"

namespace fyusion {
namespace fyusenet {
namespace cpu {
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
CPUBuffer::CPUBuffer(const CPUBufferShape& shape) : shape_(shape) {
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
    delete tiler_;
    tiler_ = nullptr;
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
    if (!tgt) tgt = shape_.createBuffer();
    else {
        if (shape_ != tgt->shape_) THROW_EXCEPTION_ARGS(FynException,"Cannot copy buffer to incompatible target buffer");
    }
    const uint8_t * srcdata = map<uint8_t>();
    uint8_t * tgtdata = tgt->map<uint8_t>();
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
 * @return Target buffer with channel-wise data storage order
 *
 * @warning This implementation is incomplete, in its current state it can only copy channel-wise
 *          CPU buffers
 */
CPUBuffer * CPUBuffer::toChannelWise(CPUBuffer *tgt) const {
    if (!tgt) tgt = shape_.createBuffer(CPUBufferShape::order::CHANNELWISE);
    else {
        if (!tgt->shape_.sameSize(shape_) || !tgt->shape_.sameType(shape_) ||
            tgt->shape_.dataOrder() != CPUBufferShape::order::CHANNELWISE) {
            THROW_EXCEPTION_ARGS(FynException,"");
        }
    }
    if ((shape_.dataOrder_ == CPUBufferShape::order::CHANNELWISE) ||
        ((shape_.width_ == 1) && (shape_.height_ == 1))) {
        const uint8_t * srcdata = map<uint8_t>();
        uint8_t * tgtdata = tgt->map<uint8_t>();
        assert(srcdata);
        assert(tgtdata);
        memcpy(tgtdata, srcdata, bytes());
        unmap();
        tgt->unmap();
        return tgt;
    } else {
        // TODO (mw) implement missing conversions
        THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
    }
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
    if (shape_.dataOrder_ != CPUBufferShape::order::GPU_SHALLOW) {
        // TODO (mw) implement missing conversions
        THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
    }
    if (!tgt) tgt = shape_.createBuffer(CPUBufferShape::order::GPU_SHALLOW);
    else {
        if (!tgt->shape().sameOrder(shape_)) {
            THROW_EXCEPTION_ARGS(FynException,"Data order of target buffer is not compatible");
        }
    }
    const uint8_t * srcdata = map<uint8_t>();
    uint8_t * tgtdata = tgt->map<uint8_t>();
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
    if (!tgt) tgt = shape_.createBuffer(CPUBufferShape::order::GPU_DEEP);
    else {
        // check if tgt is compatible
    }
    if (shape_.dataOrder_ == CPUBufferShape::order::GPU_DEEP) {
        const uint8_t * srcdata = map<uint8_t>();
        uint8_t * tgtdata = tgt->map<uint8_t>();
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
    FILE *out = fopen(fileName,"w");
    if (!out) THROW_EXCEPTION_ARGS(FynException,"Cannot open file %s for writing", fileName);
#endif
    switch (shape_.dataOrder_) {
        case CPUBufferShape::order::CHANNELWISE:
#ifndef FYUSENET_USE_WEBGL
            fwrite(map<uint8_t>(),1,shape_.bytes(),out);
#else
            EM_ASM({window.download($0, $1, $2);}, map<uint8_t>(), shape_.bytes(), fileName);
#endif
            unmap();
            break;
        case CPUBufferShape::order::GPU_SHALLOW: {
            T * tmp = (T *)malloc(shape_.bytes(CPUBufferShape::order::CHANNELWISE));
            if (!tmp) {
                FNLOGE("Cannot allocate tmp buffer");
                throw std::bad_alloc();
            }
            const T *src = map<T>();
            shallowToChannelWise<T>(src, tmp);
            unmap();
#ifndef FYUSENET_USE_WEBGL
            fwrite(tmp,1,shape_.bytes(CPUBufferShape::order::CHANNELWISE), out);
#else
            EM_ASM({window.download($0, $1, $2);}, tmp, shape_.bytes(CPUBufferShape::order::CHANNELWISE), fileName);
#endif
            free(tmp);
            break;
        }
        case CPUBufferShape::order::GPU_DEEP: {
            T * tmp = (T *)malloc(shape_.bytes(CPUBufferShape::order::CHANNELWISE));
            if (!tmp) {
                FNLOGE("Cannot allocate tmp buffer");
                throw std::bad_alloc();
            }
            const T *src = map<T>();
            deepToChannelWise<T>(src, tmp);
            unmap();
#ifndef FYUSENET_USE_WEBGL
            fwrite(tmp,1,shape_.bytes(CPUBufferShape::order::CHANNELWISE), out);
#else
            EM_ASM({window.download($0, $1, $2);}, tmp, shape_.bytes(CPUBufferShape::order::CHANNELWISE), fileName);
#endif
            free(tmp);
            break;
        }
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
 * @retval true if read operation was succesful
 * @retval false otherwise
 *
 * This function reads the content of the supplied \p pbo into this buffer instance.
 *
 * @warning This function currently only supports \c FLOAT32 data types
 */
bool CPUBuffer::readFromPBO(opengl::PBO * pbo, CPUBufferShape::type type, uint64_t sequenceNo) {
    // TODO (mw) incorporate type parameter, by performing type conversions
    assert(type == CPUBufferShape::type::FLOAT32);
    if (!memory_) return false;
#ifdef DEBUG
    glGetError();
#endif
    pbo->bind(GL_PIXEL_PACK_BUFFER);
    uint8_t * tgt = map<uint8_t>();
    if (!tgt) THROW_EXCEPTION_ARGS(FynException,"Oops, trying to copy to an already mapped buffer");
    size_t cap = bytes();
    void * src = pbo->mapReadBuffer();
    if (!src) {
        unmap();
        THROW_EXCEPTION_ARGS(FynException,"Cannot read data from PBO");
    }
#ifdef DEBUG
    int err = glGetError();
    if (err) {
        FNLOGE("Cannot map PBO buffer, err=0x%x src=%p",err,src);
        if (src) pbo->unmapReadBuffer();
        pbo->unbind(GL_PIXEL_PACK_BUFFER);
        unmap();
        return false;
    }
#endif
    size_t sz = pbo->capacity();
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
 * @brief Translate data type of CPU buffers to OpenGL data type (not texture format)
 *
 * @param type CPU buffer data type
 *
 * @return OpenGL data type that maps to the supplied type
 *
 * @see https://www.khronos.org/opengl/wiki/Image_Format
 */
GLuint CPUBuffer::typeToGLType(CPUBufferShape::type type) {
    // TODO (mw) use LUT instead of lengthy switch/case
    GLuint gltype = 0;
    switch (type) {
        case CPUBufferShape::FLOAT32:
            gltype = GL_FLOAT;
            break;
        case CPUBufferShape::UINT32:
            gltype = GL_UNSIGNED_INT;
            break;
        case CPUBufferShape::INT32:
            gltype = GL_INT;
            break;
        case CPUBufferShape::UINT16:
            gltype = GL_UNSIGNED_SHORT;
            break;
        case CPUBufferShape::INT16:
            gltype = GL_SHORT;
            break;
        case CPUBufferShape::UINT8:
            gltype = GL_UNSIGNED_BYTE;
            break;
        case CPUBufferShape::INT8:
            gltype = GL_BYTE;
            break;
        default:
            THROW_EXCEPTION_ARGS(FynException,"Illegal data type supplied");
    }
    return gltype;
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
                                          shape_.width_, shape_.height_, shape_.channels_, shape_.channels_,
                                          1.0f, 1.0f, 0, shape_.padding_, 1, 1, 1, 1);
    }
    int lwidth = shape_.width_ + 2 * shape_.padding_;
    int lheight = shape_.height_ + 2 * shape_.padding_;
    int swidth = tiler_->getViewportWidth();
    int twidth = tiler_->getInputWidth();
    int theight = tiler_->getInputHeight();
    int channeloffset = 0;
    for (int ty=0; ty < tiler_->numOutputTiles(gpu::deep::DeepTiler::VERTICAL); ty++) {
        for (int tx=0; tx < tiler_->numOutputTiles(gpu::deep::DeepTiler::HORIZONTAL); tx++) {
            int rem = ((shape_.channels_ - channeloffset) > LayerBase::PIXEL_PACKING) ? LayerBase::PIXEL_PACKING : shape_.channels_ - channeloffset;
            const T * in = src + ((shape_.padding_ + ty*(theight+shape_.padding_))*swidth + shape_.padding_ + tx*(twidth+shape_.padding_))*LayerBase::PIXEL_PACKING;
            for (int l=0; l < rem; l++) {
                T * outptr = tgt + channeloffset * (lwidth * lheight);
                if (shape_.padding_ > 0) memset(outptr, 0, sizeof(T)*lwidth);
                for (int y=shape_.padding_; y < lheight; y++) {
                    if (shape_.padding_ > 0) outptr[y*lwidth]=0;
                    for (int x=shape_.padding_; x < lwidth; x++) {
                        outptr[x+y*lwidth] = in[(y*swidth+x)*LayerBase::PIXEL_PACKING+l];
                    }
                    if (shape_.padding_ > 0) outptr[y*lwidth+lwidth-1]=0;
                }
                if (shape_.padding_ > 0) memset(outptr+lwidth*(lheight-1),0,sizeof(T)*lwidth);
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


} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
