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
#include <cstdint>
#include <algorithm>

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


namespace internal {

    /**
     * @brief Tiling-candidate, for internal computations only
     *
     * @see CPUBufferShape::computeDeepTiling()
     */
    struct tilecand {
        tilecand(int x,int y,float cost):x(x),y(y),cost(cost) {
        }
        int x;
        int y;
        float cost;
    };

    inline static int padChannels(int channels) {
        return LayerBase::PIXEL_PACKING * ((channels + LayerBase::PIXEL_PACKING-1)/LayerBase::PIXEL_PACKING);
    }
}

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param height Tensor height
 * @param width Tensor width
 * @param channels Number of channels in the tensor
 * @param padding (Spatial) padding for the tensor
 * @param type Data type used in the tensor
 * @param order Storage order of the tensor, may either be one of the GPU orders or a simple
 *              channel-wise storage as 3D array
 *
 * Creates and initializes an object that stores the current buffer shape and data arrangement.
 */
CPUBufferShape::CPUBufferShape(int height, int width, int channels, int padding, type type, order order) :
    width_(width + 2*padding), height_(height + 2*padding), channels_(channels), padding_(padding),
      dataOrder_(order), dataType_(type) {

    if (order == order::GPU_DEEP) {
        std::pair<int,int> tiles = computeDeepTiling(channels);
        tileWidth_ = width;
        tileHeight_ = height;
        width_ = (tiles.first * width + padding) + padding;
        height_ = (tiles.second * height + padding) + padding;
        channels_ = channels;
    }
}


/**
 * @brief Destructor
 */
CPUBufferShape::~CPUBufferShape() {    
}


/**
 * @brief Comparison operator (inequality)
 *
 * @param other Object to compare with
 *
 * @retval true if objects do not store identical data
 * @retval false it objects store identical data
 */
bool CPUBufferShape::operator!=(const CPUBufferShape& other) const {
    return ! operator==(other);
}


/**
 * @brief Comparison operator (equality)
 *
 * @param other Object to compare with
 *
 * @retval true it objects store identical data
 * @retval false if objects do not store identical data
 */
bool CPUBufferShape::operator==(const CPUBufferShape& other) const {
    return sameSize(other) && sameType(other) && sameOrder(other);
    return  (width_ == other.width_) &&
            (height_ == other.height_) &&
            (channels_ == other.channels_) &&
            (dataType_ == other.dataType_) &&
            (dataOrder_ == other.dataOrder_) &&
            (padding_ == other.padding_);
}


/**
 * @brief Check if shape objects refer to the same datatype
 *
 * @param other Object to compare with
 *
 * @retval true if objects refer to the same datatype
 * @retval false otherwise
 */
bool CPUBufferShape::sameType(const CPUBufferShape& other) const {
    return (dataType_ == other.dataType_);
}


/**
 * @brief Check if shape objects refer to the same data ordering
 *
 * @param other Object to compare with
 *
 * @retval true if objects refer to the same data ordering
 * @retval false otherwise
 */

bool CPUBufferShape::sameOrder(const CPUBufferShape& other) const {
    return  (dataOrder_ == other.dataOrder_);
}



/**
 * @brief Check if two shape objects (of the same order) have the same size
 *
 * @param other Object to compare with
 *
 * @retval true if shapes encode the same size (including padding)
 * @retval false otherwise
 */
bool CPUBufferShape::sameSize(const CPUBufferShape &other) const {
    assert(dataOrder_ == other.dataOrder_);
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



/**
 * @brief Compute a new shape object in different data order
 *
 * @param newOrder Target data order to be used for the new shape object
 *
 * This function derives a new shape object from the current object in the supplied data order,
 * for example in order to create new buffers that are used for different devices.
 */
CPUBufferShape CPUBufferShape::asOrder(order newOrder) const {
    switch (dataOrder_) {
        case order::CHANNELWISE:
            return CPUBufferShape(height_ - 2 *padding_, width_ - 2 * padding_, channels_, padding_, dataType_, newOrder);
        case order::GPU_SHALLOW:
            return CPUBufferShape(height_ - 2 *padding_, width_ - 2 * padding_, channels_, padding_, dataType_, newOrder);
        case order::GPU_DEEP: {
            assert(tileWidth_ > 0);
            assert(tileHeight_ > 0);
            return CPUBufferShape(tileHeight_, tileWidth_, channels_, padding_, dataType_, newOrder);
            }
    }
    THROW_EXCEPTION_ARGS(FynException,"Cannot handle shape conversion");
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
        if (dataOrder_ == order::CHANNELWISE) {
            return width_ * height_ * channels_ * typeSize(dataType_);
        } else if (dataOrder_ == order::GPU_SHALLOW) {
            int padchans = internal::padChannels(channels_);
            return width_ * height_ * padchans * typeSize(dataType_);
        } else {
            assert(dataOrder_ == order::GPU_DEEP);
            return width_ * height_ * LayerBase::PIXEL_PACKING * typeSize(dataType_);
        }
    }
    return 0;
}



/**
 * @brief Retrieve size of the buffer in specified storage order
 *
 * @param dOrder Storage order to assume
 *
 * @return Number of bytes needed to store data from this buffer in provided storage order
 */
size_t CPUBufferShape::bytes(order dOrder) const {
    if ((width_ * height_ * channels_) > 0) {
        if (dataOrder_ == order::GPU_DEEP) {
            assert(tileWidth_ > 0);
            assert(tileHeight_ > 0);
            switch (dOrder) {
                case order::CHANNELWISE:
                    return (tileWidth_ + 2 * padding_) * (tileHeight_ + 2 * padding_) * channels_ * typeSize(dataType_);
                case order::GPU_SHALLOW: {
                    int padchans = internal::padChannels(channels_);
                    return (tileWidth_ + 2 * padding_) * (tileHeight_ + 2 * padding_) * padchans * typeSize(dataType_);
                }
                default:
                    return bytes();
            }
        } else if (dataOrder_ == order::GPU_SHALLOW) {
            switch (dOrder) {
                case order::CHANNELWISE:
                    return width_ * height_ * channels_ * typeSize(dataType_);
                case order::GPU_DEEP: {
                    std::pair<int,int> tiles = computeDeepTiling(channels_);
                    int twidth = width_ - 2 * padding_;
                    int theight = height_ - 2 * padding_;
                    int finwidth = tiles.first * (twidth + padding_) + padding_;
                    int finheight = tiles.second * (theight + padding_) + padding_;
                    return finwidth * finheight * LayerBase::PIXEL_PACKING * typeSize(dataType_);
                }
                default:
                    return bytes();
            }

        } else {
            assert(dataOrder_ == order::CHANNELWISE);
            switch (dOrder) {
                case order::GPU_SHALLOW: {
                    int padchans = internal::padChannels(channels_);
                    return width_ * height_ * padchans * typeSize(dataType_);
                }
                case order::GPU_DEEP: {
                    std::pair<int,int> tiles = computeDeepTiling(channels_);
                    int twidth = width_ - 2 * padding_;
                    int theight = height_ - 2 * padding_;
                    int finwidth = tiles.first * (twidth + padding_) + padding_;
                    int finheight = tiles.second * (theight + padding_) + padding_;
                    return finwidth * finheight * LayerBase::PIXEL_PACKING * typeSize(dataType_);
                }
                default:
                    return bytes();
            }
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


/**
 * @brief Compute tile arrangement for a given channel count
 *
 * @param channels Number of channels in the tensor
 *
 * @return X/Y tiling (width, height)
 *
 * Computes a tile arrangement that has a decent aspect ratio and does not waste too much
 * texture memory.
 */
// TODO (mw) also factor in spatial dimensions to not break texture size limits
std::pair<int,int> CPUBufferShape::computeDeepTiling(int channels) {
    // NOTE (mw) this code could use some optimization
    std::vector<internal::tilecand> candidates;
    int tiles = (channels + (LayerBase::PIXEL_PACKING-1))/LayerBase::PIXEL_PACKING;
    for (int y=1; y <= tiles; y++) {
        for (int x=y; x <= tiles; x++) {
            if (x*y >= tiles) {
                float aspectcost = (float)(((x-y) >= 0) ? (x-y) : (y-x));
                float unusedcost = (float)(x*y - tiles);
                candidates.push_back(internal::tilecand(x, y, aspectcost+unusedcost));
            }
        }
    }
    auto minitem = std::min_element(candidates.begin(),candidates.end(),[](const internal::tilecand& c1, const internal::tilecand& c2) {return c1.cost < c2.cost;});
    if (minitem == candidates.end()) {
        THROW_EXCEPTION_ARGS(FynException,"Cannot compute tiling");
    }
    return std::pair<int,int>(minitem->x,minitem->y);
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
