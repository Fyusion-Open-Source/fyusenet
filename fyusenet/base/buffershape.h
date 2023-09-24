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
namespace fyusion::fyusenet {

namespace gpu {
    class GPUBuffer;
}

namespace gpu::deep {
    class DeepTiler;
}

namespace cpu {
    class CPUBuffer;
}


/**
 * @brief Representation of tensor/buffer shapes plus some re-shaping functionality
 *
 * This class serves as shape information for the more high-level representation of tensors. It
 * was originally designed to be used in conjunction with the CPU-side representation of tensors,
 * which is not done a lot in FyuseNet and has been adapted a bit to also serve the GPU-side
 * (kind-of high-level) representation of tensors in the GPUBuffer class.
 *
 * Because FyuseNet is GPU-centric and tensors are usually represented as textures with three
 * different general formats (deep, shallow, sequence), interfacing plain and simple linear CPU
 * buffers with these more complicated layouts requires some adaptation work, which is done by this
 * class.
 *
 * As of now, many classes in FyuseNet rely on quick CPU exchange using the \e shallow GPU format,
 * which stores the channels in batches of 4. For this reason, whenever a buffer shape is encountered
 * that has more than 4 channels and is in shallow GPU format, the CPUBuffer instances have to
 * ensure that they also follow this data format.
 *
 * @note There is a high chance that shallow-type GPU buffers will be deprecated in the future, as
 *       they are not really used in more recent architectures.
 *
 * @see CPUBuffer, GPUBuffer
 */
class BufferShape {
    friend class cpu::CPUBuffer;
    friend class gpu::GPUBuffer;
 public:
    using order = BufferSpec::order;
    using type = BufferSpec::dtype;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    BufferShape(int height, int width, int channels, int padding, type type, order order = order::CHANNELWISE);
    BufferShape(int embedDim, int seqLen, type typ, int packing = gpu::PIXEL_PACKING);
    ~BufferShape() = default;

    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    bool operator!=(const BufferShape& other) const;
    bool operator==(const BufferShape& other) const;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] bool sameSize(const BufferShape &other) const;
    [[nodiscard]] bool sameType(const BufferShape &other) const;
    [[nodiscard]] bool sameOrder(const BufferShape &other) const;
    [[nodiscard]] cpu::CPUBuffer * createCPUBuffer() const;
    [[nodiscard]] cpu::CPUBuffer * createCPUBuffer(order order) const;


    [[nodiscard]] BufferShape asOrder(order newOrder) const;

    /**
     * @brief Get (native) data order for this instance
     *
     * @return order specifier with the current (native) data order
     */
    [[nodiscard]] order dataOrder() const {
      return dataOrder_;
    }

    /**
     * @brief Get (native) data type for this instance
     *
     * @return type specifier with the current (native) data type
     */
    [[nodiscard]] type dataType() const {
      return dataType_;
    }

    static size_t typeSize(type dType);
    [[nodiscard]] size_t bytes() const;
    [[nodiscard]] size_t bytes(order dOrder) const;

    template<typename T>
    cpu::CPUBuffer * cpuFromRawBuffer(const void *src, order inputOrder, int inputPadding);

    /**
     * @brief Get width of tensor
     *
     * @return Width of tensor
     *
     * @note If the data is padded, the value will include the padding for the respective data order
     */
    [[nodiscard]] int width() const {
        return width_;
    }

    /**
     * @brief Get height of tensor
     *
     * @return Height of tensor
     *
     * @note If the data is padded, the value will include the padding for the respective data order
     */
    [[nodiscard]] int height() const {
        return height_;
    }

    /**
     * @brief Get number of channels for the tensor
     *
     * @return Channels in tensor
     */
    [[nodiscard]] int channels() const {
        return channels_;
    }

    /**
     * @brief Get spatial padding on tensor borders
     *
     * @return Padding on tensor borders (always isotropic)
     */
    [[nodiscard]] int padding() const {
        return padding_;
    }


 protected:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;             //!< Width of the tensor (w/ padding)
    int height_ = 0;            //!< Height of the tensor (w/ padding)
    int channels_ = 0;          //!< Number of channels in the tensor
    int padding_ = 0;           //!< Spatial padding in the tensor
    order dataOrder_;           //!< General data order (packed GPU shallow/deep or simple channelwise representation)
    type dataType_;             //!< Data type of the tensor data (e.g. 32-bit float, 8-bit int etc.)
    int tileWidth_ = 0;         //!< For tile-based formats, stores the width of each tile (excluding padding)
    int tileHeight_ = 0;        //!< For tile-based formats, stores the height of each tile (excluding padding)
};


} // fyusion::fyusenet namespace


// vim: set expandtab ts=4 sw=4:
