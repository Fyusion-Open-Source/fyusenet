//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Neural Network Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <typeinfo>
#include <mutex>
#include <vector>
#include <cstdint>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/texture.h"
#include "../gpu/gfxcontextlink.h"
#include "../gpu/gfxcontexttracker.h"
#include "../gl/shaderprogram.h"
#include "../base/bufferspec.h"
#include "../base/layerbase.h"
#include "../common/logging.h"
#include "../base/layerflags.h"
#include "gpulayerbuilder.h"
#include "../cpu/cpubuffer.h"
#include "gpubuffer.h"
#include "rudiments/preamblegenerator.h"

//------------------------------------- Public Declarations ----------------------------------------

class LayerTestBase;

namespace fyusion {

namespace opengl {
  class FBO;
}

namespace fyusenet {
class NeuralNetwork;
class BufferManager;

namespace gpu {

/**
 * @brief GPU-specific base class for neural network layers
 *
 * This is the base class for GPU-specific neural network layers. It expands the LayerBase class
 * by a few default implementations and adds an interface for input/output textures as well
 * as interfaces to GPU specifics like FBO instances.
 *
 * Using this class as base, the system spawns into two main branches: shallow-data layers
 * and deep-data layers. The difference between shallow and deep layers is the buffer/tensor
 * layout. Shallow data layers, which are layers that typically work on less than 48 channels
 * use a multi-texture data layout, whereas deep data layers use a single texture with a different
 * layout.
 *
 * @par Shallow-Data Layers and Tensors
 * Shallow-data layers on the GPU are arranged as set of textures. Each texture can store up to
 * 4 channels such that for example an 48-channel tensor requires 12 textures for storage in total.
 * My benchmarking results on mobile GPUs (2017) have shown that using multiple render targets
 * with less render passes outperform a tiled texture format for smaller amounts of channels. The
 * individual cut-off depends a bit on the GPU, but it was usually faster in as many as 32 channels,
 * on some GPUs even up to 64 channels. As neural networks often use bottleneck-like architectures,
 * some network layers work faster with shallow data representation whereas some layers (in the
 * same network) are better be used with a different, deep-data, layout. There is a high chance
 * that shallow tensors will be deprecated in the future.
 *
 * @par Deep-Data Layers and Tensors
 * For cases where the number of channels of a tensor goes beyond a certain threshold and even up
 * into the 100s or 1000s, the multiple-render target approach becomes quite inefficient.
 * Usually, there is a negative correlation between spatial size and number of channels, such that
 * when taking the approach of tiling each 4-channel portion into patches on a larger texture,
 * a high-channel tensor can be represented with a single, tiled, texture. This is the approach
 * that is taken in that case. A tiling strategy computes the distribution of the tiles on the
 * texture, such that it fits within the maximum texture extents of the system. Padding is
 * accomplished by adding the specified amount of padding between the tiles without doubling
 * the padding on the inner tile borders.
 *
 * @par Sequence Layers and Tensors
 * Sequence layers are a special case of deep-data layers. They are used to represent sequences
 * that are based on the embedding of tokens. Opposed to shallow or deep tensors, sequence tensors
 * are \e variable in their size, namely the number of tokens. As we try to avoid re-allocations
 * of textures on a frequent basis, we instead choose to allocate a large texture that can hold
 * up to a maximum sequence length. The actual sequence length is then stored in a \c StateToken
 * which is passed to the forward() method as additional input. Sequence tensors that do not
 * reach the maximum length are only partially filled.
 *
 * @par Padding and Boundary Handling
 * For several reasons, mostly historical reasons due to the incremental "per need" basis of
 * FyuseNet's development, the padding on convolutions in FyuseNet can be confusing. For example,
 * when performing convolutions on unpadded \e shallow data, the resulting tensor is not spatially
 * shrunk but edge-clamping is used on the boundary instead. This is equivalent to infinitely
 * extending the first/last element of the affected axis on the boundary. It is even more confusing
 * on \e deep data, where the behaviour will differ depending on the channel, which is due to the
 * used tiling mechanism. The tiles that are on the texture boundary will do edge-clamping when
 * accessed past the boundary. However, read operations that leave a tile without leaving the texture
 * will spill over to neighboring tiles. For this reason it is advised to always ensure that
 * proper padding is done for convolutions. In the case of sequence tensors, padding is not
 * supported.
 *
 * @see gpu::vanilla::Shallow2DeepLayer, gpu::vanilla::Deep2ShallowLayer
 * @see gpu::deep::DeepLayerBase, gpu::deep::DeepTiler
 */
class GPULayerBase : public LayerBase, public GfxContextTracker {
    friend class ::LayerTestBase;
    friend class fyusion::fyusenet::BufferManager;
    friend class fyusion::fyusenet::NeuralNetwork;
    friend class GPUAsyncLayer;
 public:

#ifdef HIGH_PRECISION
    static constexpr BufferSpec::sizedformat TEXTURE_IFORMAT_4 = BufferSpec::sizedformat::RGBA32F;
    static constexpr BufferSpec::genericformat TEXTURE_FORMAT_4 = BufferSpec::genericformat::RGBA;
    static constexpr BufferSpec::dtype TEXTURE_TYPE_DEFAULT = BufferSpec::dtype::FLOAT;
    static constexpr opengl::Texture::pixtype TEXTURE_PIXTYPE = opengl::Texture::FLOAT32;
#else
    static constexpr BufferSpec::sizedformat TEXTURE_IFORMAT_4 = BufferSpec::sizedformat::RGBA16F;
    static constexpr BufferSpec::genericformat TEXTURE_FORMAT_4 = BufferSpec::genericformat::RGBA;
    static constexpr BufferSpec::dtype TEXTURE_TYPE_DEFAULT = BufferSpec::dtype::FLOAT16;
    static constexpr opengl::Texture::pixtype TEXTURE_PIXTYPE = opengl::Texture::FLOAT16;
#endif
    static constexpr BufferSpec::sizedformat TEXTURE_HI_IFORMAT_4 = BufferSpec::sizedformat::RGBA32F;
    static constexpr BufferSpec::dtype TEXTURE_HI_DEFAULT = BufferSpec::dtype::FLOAT32;
    static constexpr opengl::Texture::pixtype TEXTURE_HI_PIXTYPE = opengl::Texture::FLOAT32;
#ifdef FYUSENET_USE_EGL
    // OES textures (GLES only)
    static constexpr BufferSpec::sizedformat TEXTURE_IFORMAT_OES = BufferSpec::sizedformat::RGBA8;
    static constexpr BufferSpec::genericformat TEXTURE_FORMAT_OES = BufferSpec::genericformat::RGBA;
    static constexpr BufferSpec::dtype TEXTURE_TYPE_OES = BufferSpec::dtype::UBYTE;
#endif

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit GPULayerBase(const GPULayerBuilder &builder);
    GPULayerBase(const GPULayerBuilder &builder, int layerNumber);
    ~GPULayerBase() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
    [[nodiscard]] virtual GPUBuffer *getGPUOutputBuffer(int port) const;
    [[nodiscard]] virtual GPUBuffer *getGPUInputBuffer(int port) const;
    virtual void setGPUInputBuffer(GPUBuffer * buffer, int port);
    virtual void setGPUOutputBuffer(GPUBuffer * buffer, int port);
    void writeResult(const char *fileName, bool includePadding) override;
    virtual void copyResult(float *memory, bool includePadding);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] virtual int numFBOs() const;
    [[nodiscard]] virtual FBO *getFBO(int channelIndex) const;
    [[nodiscard]] virtual GLuint getOutputTexture(int channelIndex) const;
    virtual void addResidualTexture(GLuint textureID, int channelIndex);
    virtual void addResidualTexture(const Texture2D &, int channelIndex);
    virtual void addInputTexture(GLuint textureID, int channelIndex);
    virtual void addInputTexture(const Texture2D &texture, int channelIndex);
    virtual void updateInputTexture(GLuint textureID, int channelIndex);
    virtual void updateInputTexture(const Texture2D &texture, int channelIndex);
    virtual void addOutputTexture(GLuint textureID, int channelIndex, int shadowIndex);
    virtual void addOutputTexture(const Texture2D &texture, int channelIndex, int shadowIndex);
    virtual void clearInputTextures();
    virtual void clearOutputTextures();
    [[nodiscard]] virtual bool hasInputTexture(int channelIndex) const;
    [[nodiscard]] virtual GLuint getInputTexture(int channelIndex) const;
    [[nodiscard]] virtual bool hasOutputTexture(int channelIndex) const;
    void prepareRender(bool blend = true, bool depth = false, bool ignoreVP = false);
    programptr compileShaderPair(const char *vertexName, const char *fragmentName,
                                 const char *preprocDefs, const std::type_info &typeInfo);
    void disableTextureUnits(int numUnits, int startUnit = 0);
    [[nodiscard]] virtual BufferSpec::order getInputOrder(int port) const;
    [[nodiscard]] virtual BufferSpec::order getOutputOrder(int port) const;
    [[nodiscard]] virtual BufferSpec::dtype getInputType(int port) const;
    [[nodiscard]] virtual BufferSpec::dtype getOutputType(int port) const;
    static GPUBuffer * createGPUBuffer(int width, int height, int channels, BufferSpec::order order, BufferSpec::dtype type, int padding);
    static void pushSliceToBuffer(GPUBuffer *buffer, GLuint handle, int width, int height, int channels, BufferSpec::dtype type);
    static GLuint getBufferSlice(const GPUBuffer * buffer, int slice);

    /**
     * @brief Prepare/initialize set of FBOs for writing the layer results
     *
     * OpenGL requires a target framebuffer to render the data into. In order to use textures as a
     * buffer mechanism, instead of the default framebuffer (which is for example a surface that is
     * displayed on the screen), we use framebuffer objects (%FBO) that are backed by the output textures
     * to do the rendering.
     *
     * @see updateFBOs()
     */
    virtual void setupFBOs() = 0;

    /**
     * @brief Update FBO color attachments after changing the output textures
     *
     * Changing the output textures \e after initializing a network layer requires to update the
     * FBOs rendering to those textures, as they will render into the old textures otherwise.
     * This function makes sure that the FBOs in the layer are up-to-date with the current textures.
     *
     * @see #outputChanged_, setupFBOs()
     */
    virtual void updateFBOs() = 0;

    /**
     * @brief Get pointer to viewport size data
     *
     * @return Pointer to viewport size, composed of width followed by height (in pixels)
     */
    [[nodiscard]] const int *getViewport() const {
        return viewport_;
    }

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::recursive_mutex processingLock_;        //!< Lock that prevents multi-threaded calls to forward()
    std::vector<GLuint> inputTextures_;          //!< List of input textures to read data from
    std::vector<GLuint> outputTextures_;         //!< List of textures that comprise the output
    std::vector<GLuint> residualTextures_;       //!< List of textures to be added to the results of the layer op
    std::vector<FBO *> framebuffers_;            //!< List of output framebuffer objects
    int viewport_[2] = {0, 0};                   //!< Output (render) viewport size
    int residualViewport_[2] = {0, 0};             //!< Output viewport size for optional residual input
    bool outputChanged_ = false;                 //!< Indicator that an output texture has been changed (invalidates the FBOs)
    rudiments::PreambleGenerator preprocessor_;  //!< Generator for preprocessor preambles for shaders
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
