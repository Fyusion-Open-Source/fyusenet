//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Framebuffer Object Wrapper
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "fbo.h"
#include "glexception.h"
#include "../common/miscdefs.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::opengl {
const GLenum FBO::WRITE_BUFFERS[FBO::MAX_DRAWBUFFERS]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,
                                                       GL_COLOR_ATTACHMENT3,GL_COLOR_ATTACHMENT4,GL_COLOR_ATTACHMENT5,
                                                       GL_COLOR_ATTACHMENT6,GL_COLOR_ATTACHMENT7};
}

//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion {
namespace opengl {

#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
static GLenum CHANNELS_TO_FMT[4] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
static GLenum CHANNELS_TO_FMT_INT[4] = {GL_RED_INTEGER, GL_RG_INTEGER, GL_RGB_INTEGER, GL_RGBA_INTEGER};
#else
static GLenum CHANNELS_TO_FMT[4] = {GL_RED, GL_RG, GL_RGBA, GL_RGBA};
static GLenum CHANNELS_TO_FMT_INT[4] = {GL_RED_INTEGER, GL_RG_INTEGER, GL_RGBA_INTEGER, GL_RGBA_INTEGER};
#endif

/**
 * @brief Counter that keeps track of allocated (internal) texture memory consumed by the FBOs
 *
 * @note Only used in debug builds
 */
std::atomic<int64_t> FBO::textureMemory_;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Create empty %FBO (w/o backing texture)
 *
 * @param context Link to GL context
 * @param width Width for the %FBO in pixels
 * @param height Height for the %FBO in pixels
 */
FBO::FBO(const fyusenet::GfxContextLink & context, int width, int height) : GfxContextTracker(),
    width_(width), height_(height) {
    setContext(context);
}


/**
 * @brief Create %FBO with internal backing texture
 *
 * @param context Link to GL context
 * @param width Width for the %FBO in pixels
 * @param height Height for the %FBO in pixels
 * @param channels Number of channels for the %FBO (actually for the first color buffer of the %FBO)
 * @param type Pixel type for the %FBO (actually for the first color buffer of the %FBO
 * @param target Optional parameter that selects the texture target for the internal texture, default
 *               to \c GL_TEXTURE_2D
 *
 * This constructor creates an %FBO with an internally allocated backing texture according to
 * the supplied parameters. The backing texture will be used as \c GL_COLOR_ATTACHMENT0. Consider
 * this construction of an %FBO as convenient way to create a simple %FBO with only single color
 * buffer without the neeed to explicitly feed it a backing texture.
 *
 * @throws GLException if %FBO could not be generated
 */
FBO::FBO(const fyusenet::GfxContextLink & context, int width, int height, int channels, Texture::pixtype type, GLenum target) : GfxContextTracker(),
    width_(width), height_(height) {
    setContext(context);
    addTexture(GL_COLOR_ATTACHMENT0, setupInternalTexture(width, height, channels, type, target), target);
    numDrawBuffers_ = 1;
    unbind();
}

/**
 * @brief Create %FBO with single (external) color texture
 *
 * @param context Link to GL context
 * @param width Width for the %FBO in pixels
 * @param height Height for the %FBO in pixels
 * @param color0Texture OpenGL texture handle that should be used as 1st color attachment (\c COLOR0)
 * @param target Optional texture target, defaults to \c GL_TEXTURE_2D
 *
 * @throws GLException if %FBO could not be generated
 *
 * @note The supplied texture handle is not owned by this object, it is up to the caller to ensure
 *       texture resource maintenance in this case.
 */
FBO::FBO(const fyusenet::GfxContextLink & context, int width, int height, GLuint color0Texture, GLenum target) : GfxContextTracker(),
    width_(width), height_(height) {
    assert(color0Texture > 0);
    setContext(context);
    addTexture(GL_COLOR_ATTACHMENT0, color0Texture, target);
    numDrawBuffers_ = 1;
    unbind();
}


/**
 * @brief Create %FBO with single external color texture at \c GL_COLOR_ATTACHMENT0
 *
 * @param context Link to GL context
 * @param backingTexture Texture2D object that wraps the texture to be used as 1st color attachment
 *
 * @throws GLException if %FBO could not be generated
 *
 * @note The supplied texture handle is not owned by this object, it is up to the caller to ensure
 *       texture resource maintenance in this case.
 */
FBO::FBO(const fyusenet::GfxContextLink &context, const Texture2D &backingTexture) : GfxContextTracker(),
    width_(backingTexture.width()), height_(backingTexture.height()) {
    setContext(context);
    addTexture(GL_COLOR_ATTACHMENT0, backingTexture.getHandle(), backingTexture.target());
    numDrawBuffers_ = 1;
    unbind();
}


/**
 * @brief Destructor
 *
 * Deletes the %FBO and the internal backing texture (if it was set up). External textures are not
 * deallocated.
 *
 * @warning If the destructor is called with a different GL context bound, this will lead to a
 *          GL memory leak.
 */
FBO::~FBO() {
    if (context_.isCurrent()) {
        if (bound_) unbind();
        if (handle_) glDeleteFramebuffers(1, &handle_);
        if (numInternalTextures_ > 0) {
            glDeleteTextures(numInternalTextures_, internalTextures_);
#ifdef DEBUG
            for (int t=0; t < numInternalTextures_; t++) {
                textureMemory_.fetch_sub(width_ * height_ * internalChannels_[t] * Texture::channelSize(internalTypes_[t]));
            }
#endif
            numInternalTextures_ = 0;
        }
    }
}


/**
 * @brief Invalidates all framebuffer attachments
 *
 * @pre %FBO must be bound, if not, no invalidation will be done
 *
 * @throws GLException if more than 32 attachments are linked to the FBO
 */
void FBO::invalidate() {  
    if (bound_) {
        GLenum attachments[32];
        if (attachments_.size() > 32) THROW_EXCEPTION_ARGS(GLException,"Too many attachments");
        if (attachments_.size() == 0) return;
        int i=0;
        for (auto ai = attachments_.begin() ; ai != attachments_.end(); ++ai,i++) {
            attachments[i] = ai->first;
        }
#ifndef __APPLE__
        glInvalidateFramebuffer(GL_FRAMEBUFFER, attachments_.size(), attachments);
#endif
    }
    dbDirty_ = true;
}


/**
 * @brief Resize the %FBO
 *
 * @param width New width
 * @param height New height
 *
 * @pre %FBO must be bound
 */
void FBO::resize(int width, int height) {
    if (numInternalTextures_ > 0) {
        for (int t=0; t < numInternalTextures_; t++) {
            glBindTexture(internalTargets_[t], internalTextures_[t]);
#ifdef DEBUG
            int diff = width * height - width_ * height_;
            diff *= internalChannels_[t] * Texture::channelSize(internalTypes_[t]);
            textureMemory_.fetch_add(diff);
#endif
            auto ti = Texture::textureInfo(internalTypes_[t], internalChannels_[t]);
            glTexImage2D(internalTargets_[t], 0, ti.intFormat, width_, height_, 0, ti.format, ti.dataType, nullptr);
        }
    } else {
        // clear existing attachments as they cannot be used anymore, note that the framebuffer
        // will be incomplete after this operation
        for (const auto & item : attachments_) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, item.first, GL_TEXTURE_2D, item.second, 0);
        }
    }
    width_ = width;
    height_ = height;
}



/**
 * @brief Write %FBO color contents to memory
 *
 * @param memory Pointer to memory where content should be written to
 * @param channels Number of channels (must be in [1..4]) per attachment
 * @param bufsize Buffer size (in bytes)
 * @param integral Optional parameter, when set to \c true indicates that the %FBO is backed by an
 *                 \e integral texture (defaults to \c false )
 *
 * @note This is a slow operation as it uses \c glReadPixels() with a memory target and therefore
 *       has to sync the GL pipeline. This function only writes the color attachment(s) to memory,
 *       depth and stencil attachments are ignored.
 *
 * @warning In order to remain compatible with GLES and WebGL backends, make sure to round up
 *          the \p bufsize to have 4 channels whenever 3 channel RGB data is supposed to be used.
 *          GLES and therefore also WebGL cannot perform read operations on RGB textures.
 *
 * @see https://www.khronos.org/opengl/wiki/Image_Format
 */
template<typename T,GLenum dtype>
void FBO::writeToMemory(T *memory, int channels, GLsizei bufsize, bool integral) {
    assert((channels > 0) && (channels <= 4));
    GLenum format = (integral) ? CHANNELS_TO_FMT_INT[channels-1] : CHANNELS_TO_FMT[channels-1];
    CLEAR_GFXERR_DEBUG
#ifdef DEBUG
    if (!bound_) bind(GL_READ_FRAMEBUFFER, true);
#else
    if (!bound_) bind(GL_READ_FRAMEBUFFER);
#endif
    int stride = channels * sizeof(T) * width_;
    int align = 1;
    if ((stride & 1) == 0) align = 2;
    if ((stride & 3) == 0) align = 4;
    if ((stride & 7) == 0) align = 8;
#ifdef DEBUG
    GLint pbo[1];
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, pbo);
    assert(pbo[0] == 0);
#endif
    glPixelStorei(GL_PACK_ALIGNMENT, align);
    GLenum err = GL_NO_ERROR;
    for (int i=0; i < (int)attachments_.size();i++) {
        if (attachments_.count(GL_COLOR_ATTACHMENT0+i) > 0) {
            glReadBuffer(GL_COLOR_ATTACHMENT0+i);
#ifdef DEBUG
            err = glGetError();
#endif
#if !defined(__APPLE__) && !defined(FYUSENET_USE_EGL) && !defined(FYUSENET_USE_WEBGL)
            glReadnPixels(0, 0, width_, height_, format, dtype, bufsize, memory);
#else
            glReadPixels(0, 0, width_, height_, format, dtype, memory);
#endif
#ifdef DEBUG
            err = glGetError();
            if (err != GL_NO_ERROR) break;
#endif
            memory += width_ * height_ * channels;
        }
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    unbind(GL_READ_FRAMEBUFFER);
#ifdef DEBUG
    if (err != GL_NO_ERROR) {
        THROW_EXCEPTION_ARGS(GLException, "Unable to readout FBO (err=0x%X)", err);
    }
#endif
}


/**
 * @brief Copy %FBO (color-only) contents to target PBO
 *
 * @param target PBO to copy data to (not bound)
 *
 * @param dataType Pixel data type to use, e.g. \c GL_UNSIGNED_BYTE or \c GL_FLOAT
 *
 * @param channels Number of channels per color-attachment in the %FBO
 *
 * @param pboOffset Offset into the supplied PBO (in bytes) where to start writing the data to
 *
 * @param bindPBO Bind %PBO before starting copy (set to \c true if %PBO is not already bound)
 *
 * @param integral If set to \c true, will assume a download to an integral data format, default is
 *                 \c false
 *
 * @return Number of bytes read from this %FBO and all its attachments
 *
 * @throws GLException on size mismatch, unsupported datatypes, framebuffer-mismatch and GL errors
 *         for debug builds
 *
 * This function optionally binds the %FBO to the read framebuffer, binds the supplied target PBO
 * and invokes a GL read-pixel operation with the supplied PBO as target. It transfers all color
 * attachments of the %FBO to the supplied %PBO. It is recommended to use %PBO transfers using
 * a multi-threaded setup and fences.
 *
 * @warning This function assumes that this %FBO has color attachments only and that all color
 *          attachments are of RGBA type (4 channels each).
 *
 * @see GfxContextLink::issueSync(), https://www.khronos.org/opengl/wiki/Pixel_Buffer_Object
 */
size_t FBO::copyToPBO(PBO *target, GLenum dataType, int channels, size_t pboOffset, bool bindPBO, bool integral) {
    return copyToPBO(target, width_, height_, dataType, channels, pboOffset, bindPBO, integral);
}


/**
 * @brief Copy %FBO (color-only) contents to target PBO
 *
 * @param target PBO to copy data to (not bound)
 *
 * @param width Width of the (part of) the %FBO to copy
 *
 * @param height Height of the (part of) the %FBO to copy
 *
 * @param dataType Pixel data type to use, e.g. \c GL_UNSIGNED_BYTE or \c GL_FLOAT
 *
 * @param channels Number of channels per color-attachment in the %FBO
 *
 * @param pboOffset Offset into the supplied PBO (in bytes) where to start writing the data to
 *
 * @param bindPBO Bind %PBO before starting copy (set to \c true if %PBO is not already bound)
 *
 * @param integral If set to \c true, will assume a download to an integral data format, default is
 *                 \c false
 *
 * @return Number of bytes read from this %FBO and all its attachments
 *
 * @throws GLException on size mismatch, unsupported datatypes, framebuffer-mismatch and GL errors
 *         for debug builds
 *
 * This function optionally binds the %FBO to the read framebuffer, binds the supplied target PBO
 * and invokes a GL read-pixel operation with the supplied PBO as target. It transfers all color
 * attachments of the %FBO to the supplied %PBO. It is recommended to use %PBO transfers using
 * a multi-threaded setup and fences.
 *
 * @warning This function assumes that this %FBO has color attachments only and that all color
 *          attachments are of RGBA type (4 channels each).
 *
 * @see GfxContextLink::issueSync(), https://www.khronos.org/opengl/wiki/Pixel_Buffer_Object
 */
size_t FBO::copyToPBO(PBO *target, int width, int height, GLenum dataType, int channels, size_t pboOffset, bool bindPBO, bool integral) {
    if (numAttachments() > 1) THROW_EXCEPTION_ARGS(GLException,"Too many framebuffer attachments (only 1 is allowed for now)");
    CLEAR_GFXERR_DEBUG
    int mult = 1;
    switch (dataType) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
            break;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            mult = 2;
            break;
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            mult = 4;
            break;
        default:
            THROW_EXCEPTION_ARGS(GLException, "Unsupported data type");
    }
    size_t reqsize = width * height * channels * mult;
    if (target->capacity() < (pboOffset + reqsize)) THROW_EXCEPTION_ARGS(GLException, "PBO too small (required %ld bytes, got %ld)", reqsize, target->capacity());
    int stride = channels * mult * width;
    int align = 1;
    if ((stride & 1) == 0) align = 2;
    if ((stride & 3) == 0) align = 4;
    if ((stride & 7) == 0) align = 8;
    if (bindPBO) target->bind(GL_PIXEL_PACK_BUFFER);
    glPixelStorei(GL_PACK_ALIGNMENT, align);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err) THROW_EXCEPTION_ARGS(GLException, "Copy to PBO (prior buffer read) yielded error 0x%X", err);
#endif
    size_t attoffset = 0;
    GLenum format = (integral) ? CHANNELS_TO_FMT_INT[channels-1] : CHANNELS_TO_FMT[channels-1];
    for (int i=0; i < (int)attachments_.size(); i++) {
        if (attachments_.count(GL_COLOR_ATTACHMENT0+i) > 0) {
            glReadBuffer(GL_COLOR_ATTACHMENT0+i);
            glReadPixels(0, 0 , width, height, format, dataType, (GLvoid *)(attoffset + pboOffset));
            attoffset += width * height * channels * mult;
        }
    }
#ifdef DEBUG
    err = glGetError();
    if (err) THROW_EXCEPTION_ARGS(GLException, "Copy to PBO yielded error 0x%X", err);
#endif
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    if (bindPBO) target->unbind(GL_PIXEL_PACK_BUFFER);
    return attoffset;
}



/**
 * @brief Bind framebuffer object
 *
 * @param target Target to bind FBO to, defaults to \c GL_FRAMEBUFFER
 * @param statusCheck Perform status check for framebuffer completeness
 *
 * Binds the %FBO wrapped by this object to the supplied \p target.
 *
 * @throws GLException if \p statusCheck was set to \c true and the framebuffer is not complete.
 *
 * @see bindWithViewport(), unbind()
 */
void FBO::bind(GLenum target, bool statusCheck) {
#ifdef DEBUG
    assertContext();
    if (bound_) {
        FNLOGW("FBO was already bound, please check your code");
    }
#endif
    if (!handle_) THROW_EXCEPTION_ARGS(GLException, "Cannot bind uninitialized framebuffer");
    glBindFramebuffer(target, handle_);
    if (statusCheck) {
        GLenum status = glCheckFramebufferStatus(target);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            THROW_EXCEPTION_ARGS(GLException, "Trying to bind incomplete framebuffer (status=0x%X) to target 0x%x",status,target);
        }
    }
    bound_ = true;
}


/**
 * @brief Bind framebuffer object and set viewport to %FBO dimensions
 *
 * @param target Buffer target to bind %FBO to, defaults to \c GL_FRAMEBUFFER
 */
void FBO::bindWithViewport(GLenum target) {
    bind(target, true);
    glViewport(0, 0, width_, height_);
}


/**
 * @brief Unbind currently bound %FBO
 *
 * @param target Target to unbind the %FBO from, defaults to \c GL_FRAMEBUFFER
 *
 * Technically binds a zero framebuffer to the supplied \p target .
 */
void FBO::unbind(GLenum target) {
    glBindFramebuffer(target, 0);
    bound_ = false;
}

/**
 * @brief Check if this object has supplied color/depth/stencil attachment defined
 *
 * @param attachment GL enumerator for the color/depth/stencil to check, e.g. \c GL_COLOR_ATTACHMENT0
 *
 * @retval true if supplied \p attachment is defined in this object
 * @retval false otherwise
 */
bool FBO::hasAttachment(GLenum attachment) const {
    auto a = attachments_.find(attachment);
    if (a == attachments_.end()) return false;
    return true;
}


/**
 * @brief Bind a color attachment of the FBO to a texture / unit
 *
 * @param attachment %FBO attachment to bind (defaults to  \c GL_COLOR_ATTACHMENT0 )
 * @param unit Texture unit to bind texture to (defaults to \c GL_TEXTURE0 )
 * @param target Texture target (defaults to \c GL_TEXTURE_2D )
 */
void FBO::bindAttachment(GLenum attachment, GLenum unit, GLenum target) {
    GLuint t = getAttachment(attachment);
    glActiveTexture(unit);
    glBindTexture(target, t);
}


/**
 * @brief Get texture handle of specified color attachment for this %FBO
 *
 * @param attachment Color attachment to get handle for (defaults to \c GL_COLOR_ATTACHMENT0 )
 *
 * @return OpenGL texture handle
 *
 * @throws GLException if attachment was not part of this %FBO
 */
GLuint FBO::getAttachment(GLenum attachment) const {
    auto a = attachments_.find(attachment);
    if (a == attachments_.end()) THROW_EXCEPTION_ARGS(GLException,"Attachment 0x%x does not exit",attachment);
    return a->second;
}

/**
 * @brief Check if %FBO is in a valid state
 *
 * @retval true if %FBO is valid
 * @retval false otherwise
 */
bool FBO::isValid() const {
    if (!handle_) return false;
    bool wasbound = bound_;
    if (!wasbound) {
        glBindFramebuffer(GL_FRAMEBUFFER, handle_);
        bound_ = true;
    }
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    // NOTE (mw) we silently assume that the default FB was bound before calling this function
    if (!wasbound) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        bound_ = false;
    }
    if (status == GL_FRAMEBUFFER_COMPLETE) return true;
    return false;
}


/**
 * @brief Get number of color-attachments/draw-buffers attached to this %FBO
 *
 * @return Number of attachments / draw-buffers
 */
int FBO::numDrawBuffers() const {
    if (dbDirty_) {
        int buffers=0;
        for (auto fi = attachments_.begin(); fi != attachments_.end() ; ++fi) {
            if ((fi->first >= GL_COLOR_ATTACHMENT0) && (fi->first <= GL_COLOR_ATTACHMENT15)) buffers++;
        }
        if (buffers > MAX_DRAWBUFFERS) THROW_EXCEPTION_ARGS(GLException,"Maximum number of drawbuffers (%d) exceeded: %d",MAX_DRAWBUFFERS,buffers);
        numDrawBuffers_ = buffers;
        dbDirty_=false;
    }
    return numDrawBuffers_;
}


/**
 * @brief Change write mask of %FBO
 *
 * @throws GLException on GL errors for debug builds
 *
 * @pre This particular FBO is bound to the framebuffer target
 *
 * This updates the write-mask for the %FBO, such that it writes to all attachments. Use this
 * function prior to write operations if you use more than one attachment on an %FBO .
 */
void FBO::setWriteMask() const {
    assert(bound_);
    int db = numDrawBuffers();
    CLEAR_GFXERR_DEBUG
    glDrawBuffers(db, WRITE_BUFFERS);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Illegal write mask set (err=0x%X, db=%d)",err,db);
#endif
}


/**
 * @brief Update a single color texture and resize the %FBO (also binds)
 *
 * @param attachment Color attachment to update
 * @param texture Texture object which should be backing the %FBO at that color attachment
 *
 * @post Framebuffer will be bound
 */
void FBO::updateColorAttachment(GLenum attachment, const Texture2D& texture) {
    width_ = texture.width();
    height_ = texture.height();
    if (handle_ == 0) {
        addTexture(attachment, texture);
        assert(bound_);
        return;
    }
#ifdef DEBUG
    if (!context_.isCurrent()) {
        FNLOGE("Accessing FBO from wrong context");
    }
#endif
    if (!bound_) {
        glBindFramebuffer(GL_FRAMEBUFFER, handle_);
        bound_ = true;
    }
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.getHandle(),0);
    attachments_[attachment] = texture.getHandle();
}


/**
 * @brief Update a single color texture and resize the %FBO (also binds)
 *
 * @param attachment Color attachment to update
 * @param texture Texture handle
 * @param width New width for %FBO
 * @param height New height for %FBO
 *
 * @post Framebuffer will be bound
 */
void FBO::updateColorAttachment(GLenum attachment, GLuint texture, int width, int height) {
    width_ = width;
    height_ = height;
    if (handle_ == 0) {
        addTexture(attachment, texture, GL_TEXTURE_2D);
        assert(bound_);
        return;
    }
#ifdef DEBUG
    if (!context_.isCurrent()) {
        FNLOGE("Accessing FBO from wrong context");
    }
#endif
    if (!bound_) {
        glBindFramebuffer(GL_FRAMEBUFFER,handle_);
        bound_ = true;
    }
    glFramebufferTexture2D(GL_FRAMEBUFFER,attachment,GL_TEXTURE_2D,texture,0);
    attachments_[attachment]=texture;
}

/**
 * @brief Update a single color texture
 *
 * @param attachment Color attachment to update
 * @param texture Texture handle
 *
 * @post Framebuffer will be bound
 */
void FBO::updateColorAttachment(GLenum attachment, GLuint texture) {
    if (handle_ == 0) {
        addTexture(attachment, texture, GL_TEXTURE_2D);
        assert(bound_);
        numDrawBuffers_ = 1;
        return;
    }
#ifdef DEBUG
    if (!context_.isCurrent()) {
        FNLOGE("Accessing FBO from wrong context");
    }
    glGetError();
#endif
    if (!bound_) {
        glBindFramebuffer(GL_FRAMEBUFFER, handle_);
        bound_ = true;
    }
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture, 0);
    attachments_[attachment] = texture;
#ifdef DEBUG
    int err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
}


/**
 * @brief Attach texture to the %FBO
 *
 * @param attachment Enumerator for the attachment position (e.g. \c GL_COLOR_ATTACHMENT0)
 * @param texture Wrapper object for GL textures
 *
 * @throws GLException if empty texture was supplied
 *
 * @post %FBO will be bound
 *
 * @see setWriteMask()
 */
void FBO::addTexture(GLenum attachment, const Texture2D &texture) {
    if (texture.getHandle() == 0) {
        THROW_EXCEPTION_ARGS(GLException, "Empty texture supplied");
    }
    addTexture(attachment, texture.getHandle(), texture.target());
}


/**
 * @brief Convenience function to add an internal backing texture to the FBO
 *
 * @param attachment Attachment point for the new texture
 * @param pixType Pixel format for the new texture
 * @param channels Number of channels for the new texture
 * @param target Optional texture bind target, defaults to \c GL_TEXTURE_2D
 *
 * @post %FBO will be bound
 *
 * This is a simple convenience function that adds another internal texture to the FBO. Internal
 * textures are managed by the FBO itself and are bound to its lifecycle. It is recommended to
 * make use of the other addTexture() functions, in particular if you want to use texture pooling.
 */
void FBO::addTexture(GLenum attachment, int channels, Texture::pixtype pixType, GLenum target) {
    assert(channels > 0 && channels <= 4);
    GLuint tex = setupInternalTexture(width_, height_, channels, pixType, target);
    addTexture(attachment, tex, target);
}


/**
 * @brief Attach a texture to the %FBO
 *
 * @param attachment Enumerator for the attachment position (e.g. \c GL_COLOR_ATTACHMENT0)
 * @param texture OpenGL texture handle to attach
 * @param target Target type of the texture, defaults to \c GL_TEXTURE_2D
 *
 * @post %FBO will be bound
 *
 * @throws GLException on errors and incomplete framebuffers
 *
 * @see setWriteMask()
 */
void FBO::addTexture(GLenum attachment, GLuint texture, GLenum target) {
    if (texture == 0) THROW_EXCEPTION_ARGS(GLException, "Invalid texture supplied to FBO");
    if (handle_ == 0) {
        glGenFramebuffers(1, &handle_);
        if (handle_ == 0) THROW_EXCEPTION_ARGS(GLException,"Cannot generate framebuffer");
    }
    if (!bound_) {
#ifdef DEBUG
        if (!context_.isCurrent()) {
            FNLOGE("Accessing FBO from wrong context");
        }
#endif
        glBindFramebuffer(GL_FRAMEBUFFER, handle_);
        bound_ = true;
    }
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, target, texture, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        THROW_EXCEPTION_ARGS(GLException, "Framebuffer incomplete");
    }
    attachments_[attachment] = texture;
    dbDirty_ = true;
}


/**
 * @brief Attach renderbuffer to %FBO
 *
 * @param attachment Enumerator for the attachment position (e.g. \c GL_COLOR_ATTACHMENT0)
 * @param handle Handle of renderbuffer object that should be attached at specified \p attachment
 *
 * @throws GLException on errors
 *
 * @see setWriteMask()
 */
void FBO::addRenderbuffer(GLenum attachment, GLuint handle) {
    if (!handle_) {
        glGenFramebuffers(1,&handle_);
        if (!handle_) THROW_EXCEPTION_ARGS(GLException, "Cannot generate framebuffer");
    }
    if (!bound_) {
#ifdef DEBUG
        assertContext();
#endif
        glBindFramebuffer(GL_FRAMEBUFFER, handle_);
        bound_ = true;
    }
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, handle);
    attachments_[attachment] = handle;
    dbDirty_ = true;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Create and dimensionalize internal backing texture for this %FBO
 *
 * @param width Width of the texture to be created
 * @param height Height of the texture to be created
 * @param channels Number of channels per pixel
 * @param type Pixel data type
 * @param target Texture target, defaults to \c GL_TEXTURE_2D
 *
 * @return OpenGL texture handle with the backing texture
 *
 * This function is called when a "simple" FBO is to created where we do not care which texture
 * the %FBO is rendering into. It simply creates its own internal texture, which can be queried
 * externally.
 *
 * @post #internalTextures_ has a valid GL texture handle and is set to the same value as the
 *       return value
 */
GLuint FBO::setupInternalTexture(int width, int height, int channels, Texture::pixtype type, GLenum target) {
    assert((channels > 0) && (channels <= 4));
#ifdef USE_GLES
    assert(channels != 3);
#endif
#ifdef DEBUG
    glGetError();
#endif
    assert(numInternalTextures_ < MAX_INTNL_TEXTURES);
    int idx = numInternalTextures_;
    glGenTextures(1, &internalTextures_[idx]);
    if (internalTextures_[idx] == 0) THROW_EXCEPTION_ARGS(GLException,"Cannot create internal texture handle for FBO");
    numInternalTextures_++;
    internalChannels_[idx] = channels;
    internalTypes_[idx] = type;
    internalTargets_[idx] = target;
    // ------------------------------------------------------
    // Create empty texture, we default it to edge clamping
    // and nearest neighbor interpolation, though that does
    // not matter for the FBO
    // ------------------------------------------------------
    auto ti = Texture::textureInfo(internalTypes_[idx], internalChannels_[idx]);
    glBindTexture(target, internalTextures_[idx]);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(target, 0, ti.intFormat, width, height, 0, ti.format, ti.dataType, nullptr);
#ifdef DEBUG
    assert(glGetError() == GL_NO_ERROR);
    textureMemory_.fetch_add(width * height * channels * Texture::channelSize(type));
#endif
    return internalTextures_[idx];
}

template void FBO::writeToMemory<float,GL_FLOAT>(float *memory,int channels, GLsizei bufsize, bool integral);
template void FBO::writeToMemory<uint8_t,GL_UNSIGNED_BYTE>(uint8_t *memory,int channels, GLsizei bufsize, bool integral);
template void FBO::writeToMemory<uint16_t,GL_UNSIGNED_SHORT>(uint16_t *memory,int channels, GLsizei bufsize, bool integral);
template void FBO::writeToMemory<uint32_t,GL_UNSIGNED_INT>(uint32_t *memory,int channels, GLsizei bufsize, bool integral);

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
