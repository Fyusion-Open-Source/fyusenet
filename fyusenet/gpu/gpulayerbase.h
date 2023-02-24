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

#ifdef ANDROID
#include <GLES3/gl3.h>
#else
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <OpenGL/glext.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#endif

//-------------------------------------- Project  Headers ------------------------------------------

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

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {

namespace opengl {
  class FBO;
}

using namespace opengl;

namespace fyusenet {
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
 * same network) are better be used with a different, deep-data, layout.
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
 * @par Padding and Boundary Handling
 * For several reasons, mostly historical reasons due to the incremental "per need" basis of
 * FyuseNet's development, the padding on convolutions in FyuseNet can be confusing. For example,
 * when performing convolutions on unpadded \e shallow data, the resulting tensor is not spatially
 * shrunk but edge-clamping is used on the boundary instead. This is equivalent to infinitely
 * extending the first/last element of the affected axis on the boundary. It is even more confusing
 * on \e deep data, where the behaviour will differ depending on the channel, which is due to the
 * used tiling mechanism. The tiles that are on the texture boundary will do edge-clamping when
 * accessed past the boundary. However read operations that leave a tile without leaving the texture
 * will spill over to neighboring tiles. For this reason it is advised to always ensure that
 * proper padding is done for convolutions.
 *
 * @see gpu::vanilla::Shallow2DeepLayer, gpu::vanilla::Deep2ShallowLayer
 * @see gpu::deep::DeepLayerBase, gpu::deep::DeepTiler
 */
class GPULayerBase : public LayerBase, public GfxContextTracker {
 public:
    // define default texture formats for texture buffers
    static constexpr int TEXTURE_CHANNELS = 4;

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

#ifdef FYUSENET_USE_EGL
    // OES textures (GLES only)
    const static BufferSpec::sizedformat TEXTURE_IFORMAT_OES = BufferSpec::sizedformat::RGBA8;
    const static BufferSpec::genericformat TEXTURE_FORMAT_OES = BufferSpec::genericformat::RGBA;
    const static BufferSpec::dtype TEXTURE_TYPE_OES = BufferSpec::dtype::UBYTE;
#endif
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    GPULayerBase(const GPULayerBuilder& builder, int layerNumber);
    virtual ~GPULayerBase();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual int numFBOs() const;
    virtual FBO * getFBO(int channelIndex) const;
    virtual GLuint getOutputTexture(int channelIndex) const;
    virtual bool canRun() const;
    virtual void addResidualTexture(GLuint textureID, int channelIndex);
    virtual void addInputTexture(GLuint textureID, int channelIndex);
    virtual void updateInputTexture(GLuint textureID, int channelIndex);
    virtual void addOutputTexture(GLuint textureID, int channelIndex, int shadowIndex=0);
    virtual void clearInputTextures();
    virtual void clearOutputTextures();
    virtual bool hasInputTexture(int channelIndex) const;
    virtual GLuint getInputTexture(int channelIndex) const;
    virtual bool hasOutputTexture(int channelIndex) const;
    virtual void writeResult(const char *fileName, bool includePadding=false) override;
    virtual void copyResult(float * memory, bool includePadding=false);

    /**
     * @brief Get pointer to viewport size data
     *
     * @return Pointer to viewport size, composed of width followed by height (in pixels)
     *
     * @todo This function should be renamed, it is too specific to a graphics/GL context
     */
    const int * getViewport() const {
        return viewport_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    /**
     * @brief Prepare/initialize set of FBOs for writing the layer reults
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
     * This function makes sure that the FBOs in the layer are up to date with the current textures.
     *
     * @see #outputChanged_, setupFBOs()
     */
    virtual void updateFBOs() = 0;

    size_t handleActivationPreproc(layerflags flags,char *preproc,size_t maxChars);
    size_t handlePreprocFlags(layerflags flags,char *preproc,size_t maxChars);
    void prepareRender(bool blend = true, bool depth = false);
    programptr compileShaderPair(const char *vertexName, const char *fragmentName,
                                 const char *preprocDefs, const std::type_info& typeInfo);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::recursive_mutex processingLock_;        //!< Lock that prevents multi-threaded calls to forward()
    std::vector<GLuint> inputTextures_;          //!< List of input textures to read data from
    std::vector<GLuint> outputTextures_;         //!< List of textures that comprise the output
    std::vector<GLuint> residualTextures_;       //!< List of textures to be added to the results of the layer op
    std::vector<FBO *> framebuffers_;            //!< List of output framebuffer objects
    int viewport_[2] = {0, 0};                   //!< Output (render) viewport size
    int residualViewport_[2]= {0,0};             //!< Output viewport size for optional residual input
    bool outputChanged_ = false;                 //!< Indicator that an output texture has been changed (invalidates the FBOs)
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
