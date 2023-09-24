//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Framebuffer Object Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <atomic>
#include <cstdint>
#include <vector>
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "pbo.h"
#include "../gpu/gfxcontexttracker.h"
#include "../gpu/gfxcontextlink.h"
#include "texture.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::opengl {

/**
 * @brief Wrapper class for OpenGL Frame-Buffer-Object (%FBO)
 *
 * This object is a low-level / lightweight wrapper class around OpenGL framebuffer objects (FBOs).
 * It maintains the actual %FBO as well as the backing texture(s) either as external entities
 * or, for a simple %FBO case, as an internal texture.
 *
 * @see https://www.khronos.org/opengl/wiki/Framebuffer_Object
 */
class FBO : public fyusenet::GfxContextTracker {
 public:
    constexpr static int MAX_DRAWBUFFERS = 8;
    constexpr static int MAX_INTNL_TEXTURES = 8;
    // ------------------------------------------------------------------------
    // Constructors / Destructor
    // ------------------------------------------------------------------------
    FBO(const fyusenet::GfxContextLink & context, int width, int height);
    FBO(const fyusenet::GfxContextLink & context, const Texture2D& backingTexture);
    FBO(const fyusenet::GfxContextLink & context, int width, int height, int channels, Texture::pixtype type, GLenum target = GL_TEXTURE_2D);
    FBO(const fyusenet::GfxContextLink & context, int width, int height, GLuint color0Texture, GLenum target = GL_TEXTURE_2D);
    ~FBO() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] bool isValid() const;
    void invalidate();
    size_t copyToPBO(PBO *target, GLenum dataType, int channels, size_t pboOffset=0, bool bindPBO=false, bool integral=false);
    size_t copyToPBO(PBO *target, int width, int height, GLenum dataType, int channels, size_t pboOffset=0, bool bindPBO=false, bool integral=false);
    void bind(GLenum target = GL_FRAMEBUFFER, bool statusCheck=true);
    void bindWithViewport(GLenum target = GL_FRAMEBUFFER);
    void unbind(GLenum target = GL_FRAMEBUFFER);
    void addTexture(GLenum attachment, int channels, Texture::pixtype pixType, GLenum target = GL_TEXTURE_2D);
    void addTexture(GLenum attachment, GLuint handle, GLenum target = GL_TEXTURE_2D);
    void addTexture(GLenum attachment, const Texture2D& texture);
    void addRenderbuffer(GLenum attachment, GLuint handle);
    void updateColorAttachment(GLenum attachment, GLuint texture);
    void updateColorAttachment(GLenum attachment, const Texture2D& texture);
    void updateColorAttachment(GLenum attachment, GLuint texture, int width, int height);    
    void resize(int width, int height);
    [[nodiscard]] GLuint getAttachment(GLenum attachment = GL_COLOR_ATTACHMENT0) const;
    void bindAttachment(GLenum attachment = GL_COLOR_ATTACHMENT0, GLenum unit = GL_TEXTURE0, GLenum target = GL_TEXTURE_2D);
    [[nodiscard]] bool hasAttachment(GLenum attachment) const;
    [[nodiscard]] int numDrawBuffers() const;
    void setWriteMask() const;

    /**
     * @brief Retrieve wrapped OpenGL FBO handle
     *
     * @return OpenGL FBO handle or 0 if FBO is not valid
     */
    [[nodiscard]] GLuint getHandle() const {
        return handle_;
    }

    /**
     * @brief Get number of texture attachments
     *
     * @return Number of texture attachments
     */
    [[nodiscard]] int numAttachments() const {
        return attachments_.size();
    }

    /**
     * @brief Get width of the FBO
     *
     * @return FBO width in pixels
     *
     * @note Technically an FBO does not have associated dimensions, it is the textures that are
     *       backing the FBO, which have associated dimensions. For ease-of-use we are assigning
     *       dimensions to an FBO here and assume that all backing textures have the correct size.
     */
    [[nodiscard]] int width() const {
        return width_;
    }


    /**
     * @brief Get height of the FBO
     *
     * @return FBO height in pixels
     *
     * @note Technically an FBO does not have associated dimensions, it is the textures that are
     *       backing the FBO, which have associated dimensions. For ease-of-use we are assigning
     *       dimensions to an FBO here and assume that all backing textures have the correct size.
     */
    [[nodiscard]] int height() const {
        return height_;
    }

    /**
     * @brief Compute buffer size that can accommodate %FBO data
     *
     * @param channels Assumed number of channels in the %FBO
     *
     * @return Number of elements (not necessarily bytes) to allocate for a buffer
     */
    [[nodiscard]] size_t size(int channels) const {
        return width_ * height_ * channels;
    }

    /**
     * @brief Amount of texture memory consumed by all (internally backed) FBOs
     *
     * @return Size (bytes) of texture memory occupied by all internally backed FBOs
     */
    [[nodiscard]] static int64_t textureMemory() {
        return textureMemory_.load();
    }

    template<typename T, GLenum dtype>
    void writeToMemory(T *memory, int channels, GLsizei bufsize, bool integral=false);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    GLuint setupInternalTexture(int width, int height, int channels, Texture::pixtype type, GLenum target = GL_TEXTURE_2D);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    static const GLenum WRITE_BUFFERS[MAX_DRAWBUFFERS];
    int width_ = 0;                                         //!< Width of the FBO (pixels) and its backing texture(s)
    int height_ = 0;                                        //!< Height of the FBO (pixels) and its backing texture(s)
    GLuint handle_ = 0;                                     //!< FBO handle (OpenGL)
    GLuint internalTextures_[MAX_INTNL_TEXTURES];           //!< Internal textures (managed by the FBO itself)
    GLenum internalTargets_[MAX_INTNL_TEXTURES];            //!< Internal textures (managed by the FBO itself)
    uint8_t internalChannels_[MAX_INTNL_TEXTURES];          //!< Internal textures (managed by the FBO itself)
    Texture::pixtype internalTypes_[MAX_INTNL_TEXTURES];    //!< Data type for the #internalTextures_
    int numInternalTextures_ = 0;                           //!< Number of internal textures
    mutable int numDrawBuffers_ = 0;                        //!< Number of drawing buffers for this FBO
    mutable bool bound_ = false;                            //!< Indicator if FBO is currently bound
    mutable bool dbDirty_ = false;                          //!< Set to \c true if the number of draw buffers need to be recomputed
    std::unordered_map<GLint,GLuint> attachments_;          //!< Texture / Renderbuffer handles for mapped by their FBO attachment points
    static std::atomic<int64_t> textureMemory_;             //!< Tracker for internally allocated texture memory
};


} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
