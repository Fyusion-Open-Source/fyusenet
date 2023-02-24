//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Buffer Shaper
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
 * @param width Tensor width
 * @param height Tensor height
 * @param channels Number of channels in the tensor
 * @param padding (Spatial) padding for the tensor
 * @param type Data type used in the tensor
 * @param order Storage order of the tensor, may either be one of the GPU orders or a simple
 *              channel-wise storage as 3D array
 *
 * Creates and initializes an object that stores the current buffer shape and data arrangement.
 */
CPUBufferShape::CPUBufferShape(int width, int height, int channels, int padding, type type, order order) :
    width_(width), height_(height), channels_(channels), padding_(padding),
      dataOrder_(order), dataType_(type) {
}


/**
 * @brief Destructor
 */
CPUBufferShape::~CPUBufferShape() {    
}


bool CPUBufferShape::operator!=(const CPUBufferShape& other) const {
    return ! operator==(other);
}

bool CPUBufferShape::operator==(const CPUBufferShape& other) const {
    return sameSize(other) && sameType(other) && sameOrder(other);
    return  (width_ == other.width_) &&
            (height_ == other.height_) &&
            (channels_ == other.channels_) &&
            (dataType_ == other.dataType_) &&
            (dataOrder_ == other.dataOrder_) &&
            (padding_ == other.padding_);
}

bool CPUBufferShape::sameType(const CPUBufferShape& other) const {
    return (dataType_ == other.dataType_);
}

bool CPUBufferShape::sameOrder(const CPUBufferShape& other) const {
    return  (dataOrder_ == other.dataOrder_);
}


bool CPUBufferShape::sameSize(const CPUBufferShape &other) const {
    return  (width_ == other.width_) &&
            (height_ == other.height_) &&
            (channels_ == other.channels_) &&
            (padding_ == other.padding_);
}

/**
 * @brief Create new CPUBuffer instance
 *
 * @return CPUBuffer object with dimensions and data-type/order as they are stored within this
 *         shape object or \c nullptr on a zero-sized shape
 *
 * Creates a new CPU buffer, the returned buffer will have data allocated but not initialized.
 */
CPUBuffer * CPUBufferShape::createBuffer() const {
    if ((width_ * height_ * channels_) > 0) {
        return new CPUBuffer(*this);
    }
    return nullptr;
}


CPUBufferShape CPUBufferShape::asOrder(order newOrder) const {
    return CPUBufferShape(width_, height_, channels_, padding_, dataType_, newOrder);
}


/**
 * @brief Create new CPUBuffer instance
 *
 * @param order Override value for the data order
 *
 * @return CPUBuffer object with dimensions and data-type as they are stored within this shape object
 *
 * Creates a new CPU buffer, the returned buffer will have data allocated but not initialized.
 */
CPUBuffer * CPUBufferShape::createBuffer(order order) const {
    if ((width_ * height_ * channels_) > 0) {
        return new CPUBuffer(asOrder(order));
    }
    return nullptr;
}




/**
 * @brief Get element size of data type
 *
 * @param dType Data-type to check
 *
 * @return Number of bytes per element for the provided data-type
 */
size_t CPUBufferShape::typeSize(const type dType) {
    static int sizelut[NUM_TYPES] = {4,4,2,1,4,2,1};
    if (dType >= NUM_TYPES) THROW_EXCEPTION_ARGS(FynException,"Illegal type %d", (int)dType);
    return sizelut[dType];
}


/**
 * @brief Create a CPUBuffer object from a raw buffer
 *
 * @param src Pointer to raw data buffer
 * @param inputOrder Data-ordering of the buffer supplied in \p src
 * @param inputPadding Input padding of the buffer supplied in \p src
 *
 * @return CPUBuffer object that contains a copy of the data in the supplied \p src buffer
 *
 * Creates a CPUBuffer instance from a raw buffer pointer by \e copying the content of the
 * source buffer to the newly-created buffer, which dimensions are defined by this shape object.
 * This function performs a re-format of the data if required.
 *
 * @warning This function is not completely implemented, not all data/padding combinations work
 */
template<typename T>
CPUBuffer * CPUBufferShape::fromRawBuffer(const void *src, order inputOrder, int inputPadding) {
    CPUBuffer * buf = createBuffer();
    uint8_t * raw = buf->map<uint8_t>();
    assert(raw);
    // TODO (mw) add missing implementation
    if ((inputOrder == dataOrder_) && (inputPadding == padding_)) {
        memcpy(raw, src, buf->bytes());
        buf->unmap();
    } else {
        if (inputPadding != padding_) {
            THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
        }
        if (dataOrder_ == order::CHANNELWISE) {
            if (inputOrder == order::GPU_SHALLOW) buf->shallowToChannelWise((const T*)src, (T *)raw);
            else buf->deepToChannelWise((const T *)src, (T *)raw);
        } else if (dataOrder_ == order::GPU_SHALLOW) {
            THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
        } else if (dataOrder_ == order::GPU_DEEP) {
            THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
        }
        buf->unmap();
        return buf;
    }
    return nullptr;
}




/**
 * @brief Retrieve size of the buffer in bytes in its native order (e.g. shallow, deep, layerwise)
 *
 * @return Number of bytes that the buffer is wrapping
 */
size_t CPUBufferShape::bytes() const {
    if ((width_ * height_ * channels_) > 0) {
        gpu::deep::DeepTiler tiler(LayerType::DOWNLOAD,width_,height_,channels_,channels_,1.0f,1.0f,0,padding_,1,1,1,1);
        int ewidth = width_;
        int eheight = height_;
        if (dataOrder_ == order::GPU_SHALLOW || dataOrder_ == order::CHANNELWISE) ewidth = width_+2*padding_;
        else ewidth = tiler.getViewportWidth();
        if (dataOrder_ == order::GPU_SHALLOW || dataOrder_ == order::CHANNELWISE) ewidth = width_+2*padding_;
        else eheight = tiler.getViewportHeight();
        int echannels = (dataOrder_ == order::GPU_SHALLOW || dataOrder_ == order::CHANNELWISE) ? channels_ : LayerBase::PIXEL_PACKING;
        return ewidth * eheight * echannels*typeSize(dataType_);
    }
    return 0;
}



/**
 * @brief Retrieve size of the buffer in specified storage order)
 *
 * @param dOrder Storage order to assume
 *
 * @return Number of bytes needed to store data from this buffer in provided storage order
 */
size_t CPUBufferShape::bytes(order dOrder) const {
    if ((width_ * height_ * channels_) > 0) {
        gpu::deep::DeepTiler tiler(LayerType::DOWNLOAD,width_,height_,channels_,channels_,1.0f,1.0f,0,padding_,1,1,1,1);
        int ewidth = width_;
        int eheight = height_;
        if (dOrder == order::GPU_SHALLOW || dOrder == order::CHANNELWISE) ewidth = width_+2*padding_;
        else ewidth = tiler.getViewportWidth();
        if (dOrder == order::GPU_SHALLOW || dOrder == order::CHANNELWISE) ewidth = width_+2*padding_;
        else eheight = tiler.getViewportHeight();
        int echannels = (dOrder == order::GPU_SHALLOW || dOrder == order::CHANNELWISE) ? channels_ : LayerBase::PIXEL_PACKING;
        switch (dataType_) {
            case FLOAT32:
            case UINT32:
            case INT32:
                return ewidth * eheight * echannels * 4;
            case UINT16:
            case INT16:
                return ewidth * eheight * echannels * 2;
            case UINT8:
            case INT8:
                return ewidth * eheight * echannels;
            default:
                assert(false);
        }
    }
    return 0;
}



/**
 * @brief Convert OpenGL internal format to data type enumerator
 *
 * @param fmt GL internal format to convert
 *
 * @return Data type to be used for representing the supplied OpenGL format
 *
 * @see https://www.khronos.org/opengl/wiki/Image_Format
 */
CPUBufferShape::type CPUBufferShape::glToType(GLint fmt) {
    // TODO (mw) this implementation is incomplete
    switch (fmt) {
        case GL_R16F:
        case GL_RG16F:
        case GL_RGB16F:
        case GL_RGBA16F:
        case GL_R32F:
        case GL_RG32F:
        case GL_RGB32F:
        case GL_RGBA32F:
            return FLOAT32;         // we cannot handle 16-bit floats on CPU (yet), so return 32-bit
        case GL_R8:
        case GL_RG8:
        case GL_RGB8:
        case GL_RGBA8:
        case GL_R8UI:
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
            return UINT8;
        case GL_R8I:
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            return INT8;
        default:
            THROW_EXCEPTION_ARGS(FynException,"Unsupported type 0x%X supplied", fmt);
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/




/*##################################################################################################
#                 E X P L I C I T   T E M P L A T E   I N S T A N T I A T I O N S                  #
##################################################################################################*/

template CPUBuffer * CPUBufferShape::fromRawBuffer<float>(const void *src, order inputOrder, int inputPadding);
#ifndef FYUSENET_CPU_FLOAT_ONLY
template CPUBuffer * CPUBufferShape::fromRawBuffer<uint32_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * CPUBufferShape::fromRawBuffer<uint16_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * CPUBufferShape::fromRawBuffer<uint8_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * CPUBufferShape::fromRawBuffer<int32_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * CPUBufferShape::fromRawBuffer<int16_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * CPUBufferShape::fromRawBuffer<int8_t>(const void *src, order inputOrder, int inputPadding);
#endif



} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
