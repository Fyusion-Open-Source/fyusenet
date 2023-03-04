//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Buffer Shaper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cstdlib>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../common/fynexception.h"
#include "../base/bufferspec.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {

namespace gpu {
    namespace deep {
        class DeepTiler;
    }
}

namespace cpu {

class CPUBuffer;


/**
 * @brief Adapter class that stores buffer shapes and offers re-shaping functionality
 *
 * Because FyuseNet is GPU-centric and tensors are usually represented as textures with even two
 * different general formats (deep vs. shallow), interfacing this representation with a plain and
 * simple linear CPU buffer layout requires some adaptation work, which is done by this class.
 *
 * This class serves as shape information for a CPU-based tensor, which are stored as CPUBuffer
 * instances. The semantics of a CPUBuffer is stored in accompanying objects of this type and these
 * objects can also be used to create and convert CPU-side buffers from one layout into another.
 *
 * Many classes in FyuseNet rely on quick CPU exchange using the \e shallow GPU format, which
 * stores the channels in batches of 4. For this reason, whenever a buffer shape is encountered
 * that has more than 4 channels and is in shallow GPU format, the CPUBuffer instances have to
 * ensure that they also follow this data format.
 *
 * @see CPUBuffer
 */
class CPUBufferShape {
    friend class CPUBuffer;
 public:
    using order = BufferSpec::order;

     /**
     * @brief Specifier for the data type
     */
    enum type {
        FLOAT32 = 0,        //!< Data is stored as 32-bit single-precision IEEE-754 floating point
        UINT32,             //!< Data is stored as 32-bit unsigned integer
        UINT16,             //!< Data is stored as 16-bit unsigned integer
        UINT8,              //!< Data is stored as 8-bit unsigned integer
        INT32,              //!< Data is stored as 32-bit signed integer
        INT16,              //!< Data is stored as 16-bit signed integer
        INT8,               //!< Data is stored as 8-bit signed integer
        NUM_TYPES
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    CPUBufferShape(int height, int width, int channels, int padding, type type, order order = order::CHANNELWISE);
    ~CPUBufferShape();

    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    bool operator!=(const CPUBufferShape& other) const;
    bool operator==(const CPUBufferShape& other) const;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    bool sameSize(const CPUBufferShape &other) const;
    bool sameType(const CPUBufferShape &other) const;
    bool sameOrder(const CPUBufferShape &other) const;
    CPUBuffer * createBuffer() const;
    CPUBuffer * createBuffer(order order) const;


    CPUBufferShape asOrder(order newOrder) const;

    /**
     * @brief Get (native) data order for this instance
     *
     * @return order specifier with the current (native) data order
     */
    order dataOrder() const {
      return dataOrder_;
    }

    /**
     * @brief Get (native) data type for this instance
     *
     * @return type specifier with the current (native) data type
     */
    type dataType() const {
      return dataType_;
    }

    static size_t typeSize(const type dType);
    size_t bytes() const;
    size_t bytes(order dOrder) const;

    template<typename T>
    CPUBuffer * fromRawBuffer(const void *src, order inputOrder, int inputPadding);

    static type glToType(GLint fmt);

    /**
     * @brief Get width of tensor
     *
     * @return Width of tensor
     *
     * @note If the data is padded, the value will include the padding for the respective data order
     */
    int width() const {
        return width_;
    }

    /**
     * @brief Get height of tensor
     *
     * @return Height of tensor
     *
     * @note If the data is padded, the value will include the padding for the respective data order
     */
    int height() const {
        return height_;
    }

    /**
     * @brief Get number of channels for the tensor
     *
     * @return Channels in tensor
     */
    int channels() const {
        return channels_;
    }

    /**
     * @brief Get spatial padding on tensor borders
     *
     * @return Padding on tensor borders (always isotropic)
     */
    int padding() const {
        return padding_;
    }

    static std::pair<int,int> computeDeepTiling(int channels);

 protected:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;             //!< Width of the tensor (w/o padding)
    int height_ = 0;            //!< Height of the tensor (w/o padding)
    short channels_ = 0;        //!< Number of channels in the tensor
    short padding_ = 0;         //!< Spatial padding in the tensor
    order dataOrder_;           //!< General data order (packed GPU shallow/deep or simple layerwise representation)
    type dataType_;             //!< Data type of the tensor data (e.g. 32-bit float, 8-bit int etc.)
    int tileWidth_ = 0;         //!< For tile-based formats, stores the width of each tile (excluding padding)
    int tileHeight_ = 0;        //1< For tile-based formats, stores the height of each tile (excluding padding)
};


} // cpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
