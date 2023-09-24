//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// GPU Buffer Class for High-Level Usage, GL-specific code (Header)            (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <memory>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "texture.h"
#include "../base/buffershape.h"
#include "../base/layerbase.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu {

class GPULayerBase;

/**
 * @brief GPU Buffer Class for High-Level Usage
 *
 * This class provides a high-level interface to GPU buffers which can be used to get/set/update
 * GPU buffer connections on the layers during runtime. As the way buffers are handled internally
 * strongly depends on the used backend (currently OpenGL), we somewhat try to hide the details
 * in order to be able to have the same high-level code running with different low-level backends
 * in the future.
 *
 * A GPU buffer may consist of one or more "slices", which are basically 2D textures / images.
 * Whenever providing or querying slices, use the \c slice alias in this class. In any case it is
 * recommended to try to avoid the usage of slices directly as it will result in backend-specific
 * code, which might have to be adjusted for different backends in the future.
 */
class GPUBuffer {
    friend class fyusion::fyusenet::BufferShape;
    friend class fyusion::fyusenet::gpu::GPULayerBase;
    friend class fyusion::fyusenet::gpu::deep::DeepTiler;
 public:
    using slice = fyusion::opengl::Texture2D;
    using order = BufferSpec::order;
    using type = BufferShape::type;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    GPUBuffer() = default;

    static GPUBuffer * createShallowBuffer(const BufferShape & shape, bool init=false);
    static GPUBuffer * createDeepBuffer(const BufferShape &  shape, bool init=false);
    static GPUBuffer * createSequenceBuffer(const BufferShape& shape, bool init=false);
    static GPUBuffer * createShallowBuffer(const BufferShape& shape, const std::vector<slice> & textures);
    static GPUBuffer * createDeepBuffer(const BufferShape& shape, const std::vector<slice> & textures);
    static GPUBuffer * createSequenceBuffer(const BufferShape& shape, const std::vector<slice> & textures);

    /**
     * @brief Retrieve single slice from GPU buffer (backend-specific)
     *
     * @param idx Slice index
     *
     * @return Slice instance (backend-specific)
     *
     * @warning The instance returned by this function is backend-specific. For OpenGL this is a
     *          Texture2D instance, for other backends this might be different.
     */
    [[nodiscard]] slice getSlice(int idx) const {
        return textures_[idx];
    }

    /**
     * @brief Retrieve number of GPU slices in the buffer
     *
     * @return Number of slices
     */
    [[nodiscard]] int numSlices() const {
        return (int)textures_.size();
    }

    /**
     * @brief Retrieve width of GPU slices in the buffer
     *
     * @return Slice width
     */
    [[nodiscard]] int sliceWidth() const {
        return sliceWidth_;
    }

    /**
     * @brief Retrieve height of GPU slices in the buffer
     *
     * @return Slice height
     */
    [[nodiscard]] int sliceHeight() const {
        return sliceHeight_;
    }

    /**
     * @brief Reset GPU buffer to an empty buffer
     */
    void reset() {
        textures_.clear();
        width_ = 0;
        height_ = 0;
        channels_ = 0;
        padding_ = 0;
        order_ = order::CHANNELWISE;
        type_ = type::FLOAT32;
        sliceWidth_ = 0;
        sliceHeight_ = 0;
        tiles_[0] = 0;
        tiles_[1] = 0;
    }

 private:
    GPUBuffer(int width, int height, int channels, order ord, type typ, int padding=0, bool create=true, bool empty=false);
    void addTexture(GLuint handle, int width, int height, int channels, BufferSpec::dtype type);
    [[nodiscard]] GLuint getTexture(int slice) const;

    static std::pair<int,int> computeDeepTiling(int channels);

    std::vector<slice> textures_;           //!< Buffer slices (for this GL backend, slices are 2D textures)
    int width_ = 0;                         //!< Width of the buffer (not necessarily equivalent to slice width)
    int height_ = 0;                        //!< Height of the buffer (not necessarily equivalent to slice height)
    int channels_ = 0;                      //!< Number of channels in the buffer (not the slices)
    int padding_ = 0;                       //!< Spatial padding for the buffer slices (symmetric and equal for all slices)
    order order_ = order::CHANNELWISE;      //!< Data storage order for this buffer
    type type_ = type::FLOAT32;             //!< Data type of the buffer
    int sliceWidth_ = 0;                    //!< Width of each slice
    int sliceHeight_ = 0;                   //!< Height of each slice
    int tiles_[2] = {0};                    //!<
};


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:

