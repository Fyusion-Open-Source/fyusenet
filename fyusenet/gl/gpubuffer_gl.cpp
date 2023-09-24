//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// GPU Buffer Class for High-Level Usage, GL-specific code                     (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gpubuffer_gl.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------
namespace fyusion::fyusenet::gpu {

namespace {

    /**
     * @brief Tiling-candidate, for internal computations only
     *
     * @see GPUBuffer::computeDeepTiling()
     */
    struct tilecand {
        tilecand(int x,int y,float cost):x(x),y(y),cost(cost) {
        }
        int x;
        int y;
        float cost;
    };

    /**
     * @brief Convert data type to OpenGL texture pixel/texel type
     *
     * @param typ Data type to convert
     *
     * @return OpenGL texture pixel/texel type
     */
    static opengl::Texture::pixtype dataToPixType(GPUBuffer::type typ) {
        switch (typ) {
            case GPUBuffer::type::FLOAT32:
                return opengl::Texture::pixtype::FLOAT32;
            case GPUBuffer::type::FLOAT16:
                return opengl::Texture::pixtype::FLOAT16;
            case GPUBuffer::type::UINT32:
                return opengl::Texture::pixtype::UINT32_INTEGRAL;
            case GPUBuffer::type::INT32:
                return opengl::Texture::pixtype::INT32_INTEGRAL;
            case GPUBuffer::type::UINT8:
                return opengl::Texture::pixtype::UINT8;
            default:
                assert(false);
                return opengl::Texture::pixtype::INVALID;
        }
    }
}  // anonymous namespace


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Create a GPUBuffer instance that uses \c GPU_SHALLOW data ordering
 *
 * @param shape Shape object that describes the buffer
 * @param init Boolean that controls whether the buffer data should be initialized (with zeros)
 *
 * @return Pointer to created instance
 *
 * This function creates a GPUBuffer instance that uses \c GPU_SHALLOW data ordering. Depending on
 * the backend, this may influence the number of slices in the buffer. For some backends, there
 * may be no difference between \c GPU_SHALLOW and \c GPU_DEEP data ordering.
 */
GPUBuffer * GPUBuffer::createShallowBuffer(const BufferShape & shape, bool init) {
    assert(shape.dataOrder_ == order::GPU_SHALLOW);
    return new GPUBuffer(shape.width_, shape.height_, shape.channels_, order::GPU_SHALLOW, shape.dataType_, shape.padding_, init);
}


/**
 * @brief Create a GPUBuffer instance that uses \c GPU_DEEP data ordering
 *
 * @param shape Shape object that describes the buffer
 * @param init Boolean that controls whether the buffer data should be initialized (with zeros)
 *
 * @return Pointer to created instance
 *
 * This function creates a GPUBuffer instance that uses \c GPU_DEEP data ordering. Depending on
 * the backend, this may influence the number of slices in the buffer. For some backends, there
 * may be no difference between \c GPU_SALLOW and \c GPU_DEEP data ordering.
 */
GPUBuffer * GPUBuffer::createDeepBuffer(const BufferShape & shape, bool init) {
    assert(shape.dataOrder_ == order::GPU_DEEP);
    return new GPUBuffer(shape.width_, shape.height_, shape.channels_, order::GPU_DEEP, shape.dataType_, shape.padding_, init);
}


/**
 * @brief Create a GPUBuffer instance that uses \c GPU_SEQUENCE data ordering
 *
 * @param shape Shape object that describes the buffer
 * @param init Boolean that controls whether the buffer data should be initialized (with zeros)
 *
 * @return Pointer to created instance
 *
 * This function creates a GPUBuffer instance that uses \c GPU_SEQUENCE data ordering. Depending on
 * the backend, this may influence the number of slices in the buffer.
 */
GPUBuffer * GPUBuffer::createSequenceBuffer(const BufferShape & shape, bool init) {
    assert(shape.dataOrder_ == order::GPU_SEQUENCE);
    return new GPUBuffer(shape.width_, shape.height_, shape.channels_, order::GPU_SEQUENCE, shape.dataType_, 0, init);
}


/**
 * @brief Create a GPUBuffer instance that uses \c GPU_SHALLOW data ordering based around existing textures
 *
 * @param shape Shape object that describes the buffer
 * @param textures Vector of Texture2D instances that should be used as slices
 *
 * @return Pointer to created instance
 *
 * This function creates a GPUBuffer instance that uses \c GPU_SHALLOW data ordering. Depending on
 * the backend, this may influence the number of slices in the buffer.
 *
 * @warning This function is backend-specific and parameters may differ significantly between backends.
 */
GPUBuffer * GPUBuffer::createShallowBuffer(const BufferShape & shape, const std::vector<slice> & textures) {
    assert(shape.dataOrder_ == order::GPU_SHALLOW);
    assert(!textures.empty());
    auto * buffer = new GPUBuffer(shape.width_, shape.height_, shape.channels_, order::GPU_SHALLOW, shape.dataType_, shape.padding_, false, true);
    int numslices = (shape.channels_ + LayerBase::PIXEL_PACKING - 1) / LayerBase::PIXEL_PACKING;
    if (textures.size() != (size_t)numslices) THROW_EXCEPTION_ARGS(FynException, "Number of textures does not match channel count");
    for (int i = 0; i < numslices; i++) {
        for (auto & t  : textures) buffer->textures_.push_back(t);
    }
    return buffer;
}


/**
 * @brief Create a GPUBuffer instance that uses \c GPU_DEEP data ordering based around existing textures
 *
 * @param shape Shape object that describes the buffer
 * @param textures Vector of Texture2D instances that should be used as slices
 *
 * @return Pointer to created instance
 *
 * This function creates a GPUBuffer instance that uses \c GPU_DEEP data ordering. Depending on
 * the backend, this may influence the number of slices in the buffer. For some backends, there
 * may be no difference between \c GPU_SALLOW and \c GPU_DEEP data ordering.
 *
 * @warning This function is backend-specific and parameters may differ significantly between backends.
 */
GPUBuffer * GPUBuffer::createDeepBuffer(const BufferShape & shape, const std::vector<slice> & textures) {
    assert(shape.dataOrder_ == order::GPU_DEEP);
    if (textures.size() != 1) THROW_EXCEPTION_ARGS(FynException, "This function requires exactly 1 texture");
    auto * buffer = new GPUBuffer(shape.width_, shape.height_, shape.channels_, order::GPU_DEEP, shape.dataType_, shape.padding_, false, true);
    buffer->textures_.push_back(textures[0]);
    return buffer;
}


/**
 * @brief Create a GPUBuffer instance that uses \c GPU_SEQUENCE data ordering
 *
 * @param shape Shape object that describes the buffer
 * @param textures Vector of Texture2D instances that should be used as slices
 *
 * @return Pointer to created instance
 *
 * This function creates a GPUBuffer instance that uses \c GPU_SEQUENCE data ordering. Depending on
 * the backend, this may influence the number of slices in the buffer.
 *
 * @warning This function is backend-specific and parameters may differ significantly between backends.
 */
GPUBuffer * GPUBuffer::createSequenceBuffer(const BufferShape & shape, const std::vector<slice> & textures) {
    assert(shape.dataOrder_ == order::GPU_SEQUENCE);
    if (textures.size() != 1) THROW_EXCEPTION_ARGS(FynException, "This function requires exactly 1 texture");
    auto * buffer = new GPUBuffer(shape.width_, shape.height_, shape.channels_, order::GPU_SEQUENCE, shape.dataType_, 0, false, true);
    buffer->textures_.push_back(textures[0]);
    return buffer;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param width Width of the buffer, not necessarily the eventual slice width
 * @param height Height of the buffer, not necessarily the eventual slice height
 * @param channels Total number of channels ("depth") of the buffer
 * @param ord Data ordering
 * @param typ Data type
 * @param padding Spatial padding (symmetric and isotropic)
 * @param init  Initialize buffer data with zeros if \c true
 * @param empty Create empty buffer if \c true. Empty buffers do not have any textures backing them
 */
GPUBuffer::GPUBuffer(int width, int height, int channels, order ord, type typ, int padding, bool init, bool empty) :
        width_(width), height_(height), channels_(channels), padding_(padding), order_(ord), type_(typ) {
    using namespace opengl;
    switch (ord) {
        case order::GPU_SHALLOW:
            sliceWidth_ = width * 2*padding;
            sliceHeight_ = height * 2*padding;
            if (!empty) {
                int numslices = (channels + LayerBase::PIXEL_PACKING - 1) / LayerBase::PIXEL_PACKING;
                for (int i = 0; i < numslices; i++) {
                    textures_.emplace_back(sliceWidth_, sliceHeight_, dataToPixType(typ), LayerBase::PIXEL_PACKING, init);
                }
            }
            break;
        case order::GPU_DEEP: {
            auto tiling = computeDeepTiling(channels);
            tiles_[0] = tiling.first;
            tiles_[1] = tiling.second;
            sliceWidth_ = tiling.first * (width + padding) + padding;
            sliceHeight_ = tiling.second * (height + padding) + padding;
            if (!empty) textures_.emplace_back(sliceWidth_, sliceHeight_, dataToPixType(typ), LayerBase::PIXEL_PACKING, init);}
            break;
        case order::GPU_SEQUENCE:
            // width interpreted as element width (not pixels), height as max sequence length
            assert(channels > 0);
            assert(channels <= LayerBase::PIXEL_PACKING);
            assert(padding == 0);
            sliceWidth_ = (width + channels-1) / channels;
            sliceHeight_ = height;
            if (!empty) textures_.emplace_back(sliceWidth_, sliceHeight_, dataToPixType(typ), channels, init);
            break;
        case order::CHANNELWISE:
            THROW_EXCEPTION_ARGS(FynException, "Channelwise order is not supported in GL backend");
        default:
            assert(false);
    }

}


/**
 * @brief Add texture slice to buffer
 *
 * @param handle Raw OpenGL texture handle to add
 * @param width Width of the \e slice
 * @param height  Height of the \e slice
 * @param channels Number of channels (in 1..4) of the slice
 * @param type Data type stored in the slice
 */
void GPUBuffer::addTexture(GLuint handle, int width, int height, int channels, BufferSpec::dtype type) {
    assert(handle);
    auto ptype = dataToPixType(type);
    textures_.emplace_back(fyusion::opengl::Texture2DRef(handle, width, height, ptype, channels));
}


/**
 * @brief Retrieve texture slice
 *
 * @param slice Slice index of the texture to retrieve
 *
 * @return Raw OpenGL handle of the texture stored at the specified \p slice index
 */
GLuint GPUBuffer::getTexture(int slice) const {
    assert(slice < (int)textures_.size());
    return textures_[slice].getHandle();
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
std::pair<int,int> GPUBuffer::computeDeepTiling(int channels) {
    // NOTE (mw) this code could use some optimization
    std::vector<tilecand> candidates;
    int tiles = (channels + (LayerBase::PIXEL_PACKING-1))/LayerBase::PIXEL_PACKING;
    for (int y=1; y <= tiles; y++) {
        for (int x=y; x <= tiles; x++) {
            if (x*y >= tiles) {
                float aspectcost = (float)(((x-y) >= 0) ? (x-y) : (y-x));
                float unusedcost = (float)(x*y - tiles);
                candidates.emplace_back(x, y, aspectcost+unusedcost);
            }
        }
    }
    auto minitem = std::min_element(candidates.begin(),candidates.end(),[](const tilecand& c1, const tilecand& c2) {return c1.cost < c2.cost;});
    if (minitem == candidates.end()) {
        THROW_EXCEPTION_ARGS(FynException,"Cannot compute tiling");
    }
    return {minitem->x,minitem->y};
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:

