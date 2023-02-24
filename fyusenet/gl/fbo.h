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
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "pbo.h"
#include "../gpu/gfxcontexttracker.h"
#include "../gpu/gfxcontextlink.h"
#include "texture.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace opengl {

/**
 * @brief Wrapper class for OpenGL Frame-Buffer-Object (%FBO)
 *
 * This object is a low-level / lightweight wrapper class around OpenGL framebuffer objects (FBOs).
 * It maintains the actual %FBO as well as the backing texture(s) either as external entities
 * or, for a simple %FBO case, as an internal texture.
 *
 * Example usage:
 * <code
 *
 *
 * @see https://www.khronos.org/opengl/wiki/Framebuffer_Object
 */
class FBO : public fyusenet::GfxContextTracker {
 public:
    constexpr static int MAX_DRAWBUFFERS = 8;
    // ------------------------------------------------------------------------
    // Constructors / Destructor
    // ------------------------------------------------------------------------
    FBO(const fyusenet::GfxContextLink & context, int width, int height);
    FBO(const fyusenet::GfxContextLink & context, const Texture2D& backingTexture);
    FBO(const fyusenet::GfxContextLink & context, int width, int height, int channels, Texture::pixtype type, GLenum target = GL_TEXTURE_2D);
    FBO(const fyusenet::GfxContextLink & context, int width, int height, GLuint color0Texture, GLenum target = GL_TEXTURE_2D);
    virtual ~FBO();
    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    bool isValid() const;
    void invalidate();
    size_t copyToPBO(PBO *target, GLenum dataType, int channels, size_t pboOffset=0, bool bindPBO=false, bool integral=false);
    void bind(GLenum target = GL_FRAMEBUFFER, bool statusCheck=true);
    void bindWithViewport(GLenum target = GL_FRAMEBUFFER);
    void unbind(GLenum target = GL_FRAMEBUFFER);
    void addTexture(GLenum attachment, GLuint handle, GLenum target = GL_TEXTURE_2D);
    void addTexture(GLenum attachment, const Texture2D& texture);
    void addRenderbuffer(GLenum attachment, GLuint handle);
    void updateColorAttachment(GLenum attachment, GLuint texture);
    void updateColorAttachment(GLenum attachment, const Texture2D& texture);
    void updateColorAttachment(GLenum attachment, GLuint texture, int width, int height);    
    void resize(int width, int height);
    GLuint getAttachment(GLenum attachment = GL_COLOR_ATTACHMENT0) const;
    void bindAttachment(GLenum attachment = GL_COLOR_ATTACHMENT0, GLenum unit = GL_TEXTURE0, GLenum target = GL_TEXTURE_2D);
    bool hasAttachment(GLenum attachment) const;
    int numDrawBuffers() const;
    void setWriteMask() const;

    /**
     * @brief Retrieve wrapped OpenGL FBO handle
     *
     * @return OpenGL FBO handle or 0 if FBO is not valid
     */
    GLuint getHandle() const {
        return handle_;
    }

    /**
     * @brief Get number of texture attachments
     *
     * @return Number of texture attachments
     */
    int numAttachments() const {
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
    int width() const {
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
    int height() const {
        return height_;
    }

    /**
     * @brief Get GL handle of (internal) backing texture
     *
     * @return GL handle for internal backing texture, or 0 if there is no internal texture
     *
     * FBO wrappers may have an internal backing texture for convenience. To get the texture handle
     * of that type of wrappers, use this function.
     */
    GLuint getInternalTexture() const {
        return internalTexture_;
    }


    /**
     * @brief Amount of texture memory consumed by all (internally backed) FBOs
     *
     * @return Size (bytes) of texture memory occupied by all internally backed FBOs
     */
    static int64_t textureMemory() {
        return textureMemory_.load();
    }

    template<typename T, GLenum dtype>
    void writeToMemory(T *memory, int channels, GLsizei bufsize, bool integral=false);


 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    GLuint setupInternalTexture(int width,int height,int channels, Texture::pixtype type, GLenum target = GL_TEXTURE_2D);
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    static const GLenum WRITE_BUFFERS[MAX_DRAWBUFFERS];
    int width_;                                     //!< Width of the FBO (pixels) and its backing texture(s)
    int height_;                                    //!< Height of the FBO (pixels) and its backing texture(s)
    GLuint handle_ = 0;                             //!< FBO handle (OpenGL)
    GLuint internalTexture_ = 0;                    //!< Texture handle of optional internal backing texture
    GLenum internalTarget_ = GL_TEXTURE_2D;         //!< Texture target for the internal texture
    int internalChannels_;                          //!< Color channels for the #internalTexture_
    Texture::pixtype internalType_;                 //!< Data type for the #internalTexture_
    GLint internalFormat_ = 0;                      //!< GL (internal) texture format, relates to #internalTexture_ when valid
    mutable int numDrawBuffers_ = 0;                //!< Number of drawing buffers for this FBO
    mutable bool bound_ = false;                    //!< Indicator if FBO is currently bound
    mutable bool dbDirty_ = false;                  //!<
    std::unordered_map<GLint,GLenum> attachments_;  //!<
    static std::atomic<int64_t> textureMemory_;     //!< Tracker for internally allocated texture memory
};


} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
