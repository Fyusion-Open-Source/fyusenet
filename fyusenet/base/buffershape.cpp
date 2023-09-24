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

#include "../gl/gl_sys.h"
#include "../gl/pbo.h"
#include "../gpu/deep/deeptiler.h"
#include "../base/layerbase.h"
#include "../common/logging.h"
#include "../cpu/cpubuffer.h"
#include "../gpu/gpubuffer.h"

namespace fyusion::fyusenet {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


namespace internal {

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
BufferShape::BufferShape(int height, int width, int channels, int padding, type type, order order) :
    width_(width + 2*padding), height_(height + 2*padding), channels_(channels), padding_(padding),
      dataOrder_(order), dataType_(type) {

    if (order == order::GPU_DEEP) {
        std::pair<int,int> tiles = gpu::GPUBuffer::computeDeepTiling(channels);
        tileWidth_ = width;
        tileHeight_ = height;
        width_ = (tiles.first * width + padding) + padding;
        height_ = (tiles.second * height + padding) + padding;
        channels_ = channels;
    }
}

/**
 * @brief Create a shape object for a sequence buffer
 *
 * @param embedDim Embedding dimension (number of channels) for each sequence element
 * @param seqLen Length of the sequence
 * @param typ Data type used in the tensor
 * @param packing Optional channel packing (defaults to 4 elements per pixel)
 */
BufferShape::BufferShape(int embedDim, int seqLen, type typ, int packing) :
    width_(embedDim), height_(seqLen), channels_(packing), padding_(0),
      dataOrder_(order::GPU_SEQUENCE), dataType_(typ) {
}


/**
 * @brief Comparison operator (inequality)
 *
 * @param other Object to compare with
 *
 * @retval true if objects do not store identical data
 * @retval false it objects store identical data
 */
bool BufferShape::operator!=(const BufferShape& other) const {
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
bool BufferShape::operator==(const BufferShape& other) const {
    return sameSize(other) && sameType(other) && sameOrder(other);
}


/**
 * @brief Check if shape objects refer to the same datatype
 *
 * @param other Object to compare with
 *
 * @retval true if objects refer to the same datatype
 * @retval false otherwise
 */
bool BufferShape::sameType(const BufferShape& other) const {
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

bool BufferShape::sameOrder(const BufferShape& other) const {
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
bool BufferShape::sameSize(const BufferShape &other) const {
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
 * Ownership is transferred to the caller.
 */
CPUBuffer * BufferShape::createCPUBuffer() const {
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
BufferShape BufferShape::asOrder(order newOrder) const {
    switch (dataOrder_) {
        case order::CHANNELWISE:
            // intentional fallthrough
        case order::GPU_SHALLOW:
            return {height_ - 2 * padding_, width_ - 2 * padding_, channels_, padding_, dataType_, newOrder};
        case order::GPU_DEEP: {
            assert(tileWidth_ > 0);
            assert(tileHeight_ > 0);
            return {tileHeight_, tileWidth_, channels_, padding_, dataType_, newOrder};
        }
        default:
            THROW_EXCEPTION_ARGS(FynException, "Order is not supported yet");
    }
}


/**
 * @brief Create new CPUBuffer instance
 *
 * @param order Override value for the data order
 *
 * @return CPUBuffer object with dimensions and data-type as they are stored within this shape object
 *
 * Creates a new CPU buffer, the returned buffer will have data allocated but not initialized.
 * Ownership is transferred to the caller.
 */
CPUBuffer * BufferShape::createCPUBuffer(order order) const {
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
size_t BufferShape::typeSize(const type dType) {
    return BufferSpec::typeSize(dType);
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
CPUBuffer * BufferShape::cpuFromRawBuffer(const void *src, order inputOrder, int inputPadding) {
    CPUBuffer * buf = createCPUBuffer();
    // TODO (mw) be more defensive here
    assert(buf);
    auto * raw = buf->map<uint8_t>();
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
size_t BufferShape::bytes() const {
    if ((width_ * height_ * channels_) > 0) {
        switch (dataOrder_) {
            case order::CHANNELWISE:
                return width_ * height_ * channels_ * typeSize(dataType_);
            case order::GPU_SHALLOW:
                return width_ * height_ * internal::padChannels(channels_) * typeSize(dataType_);
            case order::GPU_DEEP:
                return width_ * height_ * LayerBase::PIXEL_PACKING * typeSize(dataType_);
            case order::GPU_SEQUENCE:
                return width_ * height_ * typeSize(dataType_);
            default:
                THROW_EXCEPTION_ARGS(FynException,"Order is not supported yet");
        }
    }
    return 0;
}



/**
 * @brief Retrieve size of the current buffer in specified storage order
 *
 * @param dOrder Storage order to assume
 *
 * @return Number of bytes needed to store data from this buffer in provided storage order
 */
size_t BufferShape::bytes(order dOrder) const {
    // TODO (mw) refactor this, it's even messier than the rest of the code
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
                case order::GPU_SEQUENCE: {
                    // NOTE (mw) as of now this conversion does not make a lot of sense
                    return width_ * height_ * typeSize(dataType_);
                }
                default:
                    return bytes();
            }
        } else if (dataOrder_ == order::GPU_SHALLOW) {
            switch (dOrder) {
                case order::CHANNELWISE:
                    return width_ * height_ * channels_ * typeSize(dataType_);
                case order::GPU_DEEP: {
                    std::pair<int,int> tiles = gpu::GPUBuffer::computeDeepTiling(channels_);
                    int twidth = width_ - 2 * padding_;
                    int theight = height_ - 2 * padding_;
                    int finwidth = tiles.first * (twidth + padding_) + padding_;
                    int finheight = tiles.second * (theight + padding_) + padding_;
                    return finwidth * finheight * LayerBase::PIXEL_PACKING * typeSize(dataType_);
                }
                case order::GPU_SEQUENCE: {
                    // NOTE (mw) as of now this conversion does not make a lot of sense
                    return width_ * height_ * typeSize(dataType_);
                }
                default:
                    return bytes();
            }
        } else if (dataOrder_ == order::GPU_SEQUENCE) {
            // NOTE (mw) this conversion currently does not make sense
            THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
        } else {
            assert(dataOrder_ == order::CHANNELWISE);
            switch (dOrder) {
                case order::GPU_SHALLOW: {
                    int padchans = internal::padChannels(channels_);
                    return width_ * height_ * padchans * typeSize(dataType_);
                }
                case order::GPU_DEEP: {
                    std::pair<int,int> tiles = gpu::GPUBuffer::computeDeepTiling(channels_);
                    int twidth = width_ - 2 * padding_;
                    int theight = height_ - 2 * padding_;
                    int finwidth = tiles.first * (twidth + padding_) + padding_;
                    int finheight = tiles.second * (theight + padding_) + padding_;
                    return finwidth * finheight * LayerBase::PIXEL_PACKING * typeSize(dataType_);
                }
                case order::GPU_SEQUENCE: {
                    // NOTE (mw) as of now this conversion does not make a lot of sense
                    return width_ * height_ * typeSize(dataType_);
                }
                default:
                    return bytes();
            }
        }
    }
    return 0;
}



/*##################################################################################################
#                 E X P L I C I T   T E M P L A T E   I N S T A N T I A T I O N S                  #
##################################################################################################*/


template CPUBuffer * BufferShape::cpuFromRawBuffer<float>(const void *src, order inputOrder, int inputPadding);
#ifndef FYUSENET_CPU_FLOAT_ONLY
template CPUBuffer * BufferShape::cpuFromRawBuffer<uint32_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * BufferShape::cpuFromRawBuffer<uint16_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * BufferShape::cpuFromRawBuffer<uint8_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * BufferShape::cpuFromRawBuffer<int32_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * BufferShape::cpuFromRawBuffer<int16_t>(const void *src, order inputOrder, int inputPadding);
template CPUBuffer * BufferShape::cpuFromRawBuffer<int8_t>(const void *src, order inputOrder, int inputPadding);
#endif

} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
