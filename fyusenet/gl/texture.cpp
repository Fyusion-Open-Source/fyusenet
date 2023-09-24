//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Texture Wrapper
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "glexception.h"
#include "texture.h"
#include "fbo.h"
#include "scoped_texturepool.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::opengl {

//-------------------------------------- Local Definitions -----------------------------------------

static const GLenum texfmt_[4] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
static const GLenum texfmtI_[4] = {GL_RED_INTEGER, GL_RG_INTEGER, GL_RGB_INTEGER, GL_RGBA_INTEGER};
#ifndef FYUSENET_USE_EGL
static const GLint intfmtF32_[4] = {GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F};
static const GLint intfmtF16_[4] = {GL_R16F, GL_RG16F, GL_RGB16F, GL_RGBA16F};
static const GLint intfmtU8_[4] = {GL_R8, GL_RG8, GL_RGB8, GL_RGBA8};
static const GLint intfmtUI8_[4] = {GL_R8UI, GL_RG8UI, GL_RGB8UI, GL_RGBA8UI};
#ifdef GL_R16
static const GLint intfmtU16_[4] = {GL_R16, GL_RG16, GL_RGB16, GL_RGBA16};
#endif
static const GLint intfmtUI16_[4] = {GL_R16UI, GL_RG16UI, GL_RGB16UI, GL_RGBA16UI};
static const GLint intfmtI16_[4] = {GL_R16I, GL_RG16I, GL_RGB16I, GL_RGBA16I};
static const GLint intfmtU32_[4] = {GL_R32UI, GL_RG32UI, GL_RGB32UI, GL_RGBA32UI};
static const GLint intfmtI32_[4] = {GL_R32I, GL_RG32I, GL_RGB32I, GL_RGBA32I};
#else
static const GLint intfmtF32_[4] = {GL_R32F, GL_RG32F, GL_RGBA32F, GL_RGBA32F};
static const GLint intfmtF16_[4] = {GL_R16F, GL_RG16F, GL_RGBA16F, GL_RGBA16F};
static const GLint intfmtU8_[4] = {GL_R8, GL_RG8, GL_RGBA8, GL_RGBA8};
static const GLint intfmtUI8_[4] = {GL_R8UI, GL_RG8UI, GL_RGBA8UI, GL_RGBA8UI};
#ifdef GL_R16
static const GLint intfmtU16_[4] = {GL_R16, GL_RG16, GL_RGBA16, GL_RGBA16};
#endif
static const GLint intfmtUI16_[4] = {GL_R16UI, GL_RG16UI, GL_RGBA16UI, GL_RGBA16UI};
static const GLint intfmtI16_[4] = {GL_R16I, GL_RG16I, GL_RGBA16I, GL_RGBA16I};
static const GLint intfmtU32_[4] = {GL_R32UI, GL_RG32UI, GL_RGBA32UI, GL_RGBA32UI};
static const GLint intfmtI32_[4] = {GL_R32I, GL_RG32I, GL_RGBA32I, GL_RGBA32I};
#endif


std::atomic<int64_t> Texture::allocTextureMemory_;


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Overloaded assignment operator
 *
 * @param src Object which contents should be assigned to the current ont
 *
 * @return Reference to target object after assignment
 */
Texture & Texture::operator=(const Texture& src) {
    if (this == &src) return *this;
    if (fromPool_) fromPool_->releaseTexture(handle_);
    handle_ = src.handle_;
    handleOwned_ = src.handleOwned_;
    paramPending_ = src.paramPending_;
    fromPool_ = src.fromPool_;
    target_ = src.target_;
    channels_ = src.channels_;
    dataType_ = src.dataType_;
    wantsFence_ = src.wantsFence_;
    syncID_ = src.syncID_;
    return *this;
}

void Texture::setTarget(GLenum target) {
    target_ = target;
}


bool Texture::isFloat() const {
    switch (dataType_) {
       case FLOAT16:
       case FLOAT32:
          return true;
       default:
          return false;
    }
}

/**
 * @brief Check if texture uses an integral data type (i.e. not normalized)
 *
 * @retval true if texture uses integral data type
 * @retval false otherwise
 */
bool Texture::isIntegral() const {
    switch (dataType_) {
       case UINT8_INTEGRAL:
       case UINT16_INTEGRAL:
       case INT16_INTEGRAL:
          return true;
       default:
          return false;
    }
}


/**
 * @brief Retrieve compound texture info from pixel type and channels
 *
 * @param type Pixel type
 * @param channels Number of channels
 *
 * @return Compound texinfo structure that features internal format and GL types
 */
Texture::texinfo Texture::textureInfo(pixtype type, int channels) {
    assert(channels > 0);
    assert(channels <= 4);
    GLint ifmt = intfmtU8_[channels-1];
    GLenum tt = GL_UNSIGNED_BYTE;
    GLenum fmt = texfmt_[channels-1];
    switch (type) {
        case UINT8_INTEGRAL:
            ifmt = intfmtUI8_[channels-1];
            fmt = texfmtI_[channels-1];
            break;
        case UINT16:
#ifdef GL_R16
            ifmt = intfmtU16_[channels-1];
#else
            // FIXME (mw) check if this substitute really works (GLES does not support GL_R16)
            // better yet: scratch that completely and allow only integral types for this
            ifmt = intfmtF16_[channels-1];
#endif
            tt = GL_UNSIGNED_SHORT;
            break;
        case UINT32:
            ifmt = intfmtU32_[channels-1];
            tt = GL_UNSIGNED_INT;
            break;
        case UINT32_INTEGRAL:
            ifmt = intfmtU32_[channels-1];
            fmt = texfmtI_[channels-1];
            tt = GL_UNSIGNED_INT;
            break;
        case INT32:
            ifmt = intfmtI32_[channels-1];
            tt = GL_INT;
            break;
        case INT32_INTEGRAL:
            ifmt = intfmtI32_[channels-1];
            fmt = texfmtI_[channels-1];
            tt = GL_INT;
            break;
        case UINT16_INTEGRAL:
            ifmt = intfmtUI16_[channels-1];
            fmt = texfmtI_[channels-1];
            tt = GL_UNSIGNED_SHORT;
            break;
        case INT16_INTEGRAL:
            ifmt = intfmtI16_[channels-1];
            fmt = texfmtI_[channels-1];
            tt = GL_SHORT;
            break;
        case FLOAT16:
            ifmt = intfmtF16_[channels-1];
            tt = GL_HALF_FLOAT;
            break;
        case FLOAT32:
            ifmt = intfmtF32_[channels-1];
            tt = GL_FLOAT;
            break;
        default:
            break;
    }
    return texinfo(ifmt, fmt, tt, type);
}


/**
 * @brief Retrieve size of used texture memory (w/o pooling) used by Texture objects
 *
 * @return Number of bytes of texture memory used by instances derived from Texture
 *
 * @note Debugging only, this does not account for pooling and it also does not account for
 *       raw GL textures that are not controlled by the Texture class or its derivatives.
 */
int64_t Texture::usedTextureMemory() {
    return allocTextureMemory_.load();
}


/**
 * @brief Create undefined/empty 2D texture
 */
Texture2D::Texture2D() : Texture(GL_TEXTURE_2D) {
}

/**
 * @brief Create (empty but dimensionalized) 2D texture
 *
 * @param width Width of the texture in pixels
 * @param height Height of texture in pixels
 * @param type Pixel data type to use, see pixtype
 * @param channels Number of channels per pixel
 * @param clear Optional flag, if set to true, texture is immediately cleared by performing null upload
 */
Texture2D::Texture2D(int width, int height, pixtype type, int channels, bool clear) : Texture(GL_TEXTURE_2D, channels, type),
    width_(width), height_(height) {
    assert(width > 0);
    assert(height > 0);
    assert(channels > 0);
    createHandle();
    assert(*(handle_.get()) != 0);
    glBindTexture(GL_TEXTURE_2D, *(handle_));
    updateParams();
    if (clear) this->clear();
#ifdef DEBUG
    allocTextureMemory_.fetch_add(size());
#endif
}

/**
 * @brief Create dimensionalized 2D texture from texture pool
 *
 * @param width Width of texture in pixels
 * @param height Height of texture in pixels
 * @param type Pixel data type to use, see pixtype
 * @param channels Number of channels per pixel
 * @param pool ScopedTexturePool instance to use, may be \c nullptr
 * @param scope Scope identifier for the texture pool
 * @param lock Optional parameter, if set to \c true, the texture will be locked in the pool
 *
 * Creates a 2D texture by first trying to obtain an already existing texture that matches the
 * query from the pool and only then creating a new one. In case an existing texture is used,
 * the contents of the texture are undefined, but the texture is always dimensionalized. If
 * a new texture is created, it is automatically cleared/dimensionalized.
 *
 * In case a \c nullptr is supplied as \p pool, this function falls back to the default functionality
 * of this texture wrapper and creates a new (unpooled) texture.
 *
 * @see BasicTexturePool::lock(), BasicTexturePool::unlock()
 *
 * @post The texture is bound to the \c GL_TEXTURE_2D target of the current unit
 */
Texture2D::Texture2D(int width, int height, pixtype type, int channels, ScopedTexturePool *pool, uint32_t scope, bool lock) :
    Texture(GL_TEXTURE_2D, channels, type),
    width_(width), height_(height) {
    assert(width > 0);
    assert(height > 0);
    assert(channels > 0);
    if (pool) {
        handle_ = pool->obtainTexture(width, height, channels, type, scope, lock);
        fromPool_ = pool;
        assert(*(handle_) != 0);
        glBindTexture(GL_TEXTURE_2D, *handle_);
        updateParams();
    } else {
        createHandle();
        assert(*(handle_) != 0);
        glBindTexture(GL_TEXTURE_2D, *handle_);
        updateParams();
        clear();
#ifdef DEBUG
        allocTextureMemory_.fetch_add(size());
#endif
    }
}


/**
 * @brief Destructor
 *
 * Decreases reference count on non-pool texture handles and releases texture to the pool if pool
 * texture.
 */
Texture2D::~Texture2D() {
#ifdef DEBUG
    if ((handle_.unique()) && (!fromPool_) && (handleOwned_)) {
        allocTextureMemory_.fetch_sub(size());
    }
#endif
    if (fromPool_) {
        fromPool_->releaseTexture(handle_);
    }
}


/**
 * @brief Overloaded assignment operator
 *
 * @param src Object which contents should be assigned to the current ont
 *
 * @return Reference to target object after assignment
 */
Texture2D & Texture2D::operator=(const Texture2D& src) {
    if (this == &src) return *this;
#ifdef DEBUG
    if (handle_.unique() && !fromPool_ && handleOwned_) {
        allocTextureMemory_.fetch_sub(size());
    }
#endif    
    Texture::operator=(src);
    width_ = src.width_;
    height_ = src.height_;
    memcpy(wrapMode_, src.wrapMode_, sizeof(wrapMode_));
    memcpy(interpolation_, src.interpolation_, sizeof(interpolation_));
    return *this;
}


/**
 * @brief Force texture invalidation
 *
 * For non-pooled textures, this resets the (shared) handle pointer and for pooled textures, the
 * texture is released back into the pool.
 */
void Texture2D::reset() {
#ifdef DEBUG
    if (handle_.unique() && !fromPool_ && handleOwned_) {
        allocTextureMemory_.fetch_sub(size());
    }
#endif
    if (fromPool_) {
        fromPool_->releaseTexture(handle_);
        fromPool_ = nullptr;
    }
    Texture::reset();
}


/**
 * @brief Retrieve texture size
 *
 * @return Size of texture (in bytes)
 */
uint32_t Texture2D::size() const {
    return width_ * height_ * channels_ * channelSize(dataType_);
}


/**
 * @brief Set texture wrap mode (u and v separately)
 *
 * @param uWrap Wrap mode for u-axis (also named s-axis), e.g. EDGE_CLAMP
 * @param vWrap Wrap mode for v-axis (also named t-axis), e.g. EDGE_CLAMP
 */
void Texture2D::wrapMode(wrap uWrap, wrap vWrap) {
    wrapMode_[0] = uWrap;
    wrapMode_[1] = vWrap;
    paramPending_ = true;
}


/**
 * @brief Set interpolation mode
 *
 * @param minIntp Interpolation to use on minifying texture (zooming out), e.g. NEAREST
 * @param magIntp  Interpolation to use on magnifying texture (zooming in), e.g. NEAREST
 */
void Texture2D::interpolation(intp minIntp, intp magIntp) {
    interpolation_[0] = minIntp;
    interpolation_[1] = magIntp;
    paramPending_ = true;
}


/**
 * @brief Explicitly unbind texture from specified texture unit
 *
 * @param unit Texture unit index to unbind texture from (e.g. 0)
 *
 * @note Do not pass the GL constants for texture units here (like \c GL_TEXTURE0)
 */
void Texture2D::unbind(int unit) const {
    assert(unit >= 0);
    if (unit >= 0) glActiveTexture(GL_TEXTURE0+unit);
    glBindTexture(GL_TEXTURE_2D, 0);
}


/**
 * @brief Bind texture from specified texture unit
 *
 * @param unit Texture unit index to bind texture to (e.g. 0)
 *
 * @note Do not pass the GL constants for texture units here (like \c GL_TEXTURE0)
 */
void Texture2D::bind(int unit) const {
    assert(unit >= 0);
    if (unit >= 0) glActiveTexture(GL_TEXTURE0+unit);
    glBindTexture(GL_TEXTURE_2D, *(handle_));
    if ((paramPending_) || (fromPool_)) {
        updateParams();
        paramPending_ = false;
    }
}


/**
 * @brief Upload data to texture or dimensionalize texture
 *
 * @param data Pointer to data to be uploaded to texture memory
 *
 * @pre Texture is bound to the \c GL_TEXTURE_2D target
 *
 * This function simply uploads the data supplied in \p data to the GL driver which is then
 * uploaded to texture memory. In case a \c nullptr is supplied to \p data, nothing will be
 * uploaded and instead an empty texture of the specified dimensions will be created.
 *
 *
 * @warning This function assumes that the supplied data pointer has exactly the same data type
 *          as the target GL texture. If you want more fine-grained control, see the overloaded
 *          functions.
 *
 * @see clear()
 */
void Texture2D::upload(const void *data) {
    assert(channels_ > 0);
    assert(channels_ <= 4);
    texinfo info = textureInfo(dataType_, channels_);
    upload(data, info.intFormat, info.format, info.dataType);
}


/**
 * @brief Upload image data to GPU
 *
 * @param data Pointer to image data to upload
 * @param cpuDataFmt Data format of the CPU data
 *
 * @pre Texture is bound to the \c GL_TEXTURE_2D target
 */
void Texture2D::upload(const void *data, pixtype cpuDataFmt) {
    assert(channels_ > 0);
    assert(channels_ <= 4);
    texinfo info = textureInfo(cpuDataFmt, channels_);
    upload(data, info.intFormat, info.format, info.dataType);
}


/**
 * @brief Upload data to texture or dimensionalize texture
 *
 * @param data Pointer to data to be uploaded to texture memory
 * @param internal Internal representation of the texture
 * @param format Format to be uploaded
 * @param type Data type to be uploaded (as on the CPU)
 *
 * @pre Texture is bound to the \c GL_TEXTURE_2D target
 *
 * This function simply uploads the data supplied in \p data to the GL driver which is then
 * uploaded to texture memory. In case a \c nullptr is supplied to \p data, nothing will be
 * uploaded and instead an empty texture of the specified dimensions will be created.
 *
 * @see clear()
 */
void Texture2D::upload(const void *data, GLint internal, GLenum format, GLenum type) {
#ifdef DEBUG
    glGetError();
#endif
    if (internal != internal_) {
        glTexImage2D(GL_TEXTURE_2D, 0, internal, width_, height_, 0, format, type, data);
        internal_ = internal;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, format, type, data);
    }
#ifdef DEBUG
    int err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
}


/**
 * @brief Clear texture memory on GPU by uploading null data
 */
void Texture2D::clear() {
    upload(nullptr);
}


/**
 * @brief Download texture from GPU
 *
 * @param target Pointer to target memory to write texture data to
 *
 * @note This function is supposed to be mainly used for debugging. It is not properly optimized
 *       and creates temporary FBOs for downloading
 */
template<typename T>
void Texture2D::download(T *target) {
    assert(target);
    assert(channels_ > 0);
    assert(channels_ <= 4);
    assert(handle_.get());
#if !defined(FYUSENET_USE_EGL) && !defined(FYUSENET_USE_WEBGL)
    GLenum tt = (dataType_ == UINT8) ? GL_UNSIGNED_BYTE : GL_FLOAT;
    GLenum fmt = texfmt_[channels_-1];
    glBindTexture(GL_TEXTURE_2D, *(handle_));
    glGetTexImage(GL_TEXTURE_2D, 0, fmt, tt, target);
#else
    FBO tmp(fyusenet::GfxContextLink(), width_, height_);
    tmp.addTexture(GL_COLOR_ATTACHMENT0, *(handle_), GL_TEXTURE_2D);
    if (dataType_ == UINT8) {
        tmp.writeToMemory<uint8_t,GL_UNSIGNED_BYTE>((uint8_t *)target, channels_, channels_*width_*height_);
    } else {
        tmp.writeToMemory<float,GL_FLOAT>((float *)target, channels_, channels_*width_*height_*sizeof(float));
    }
#endif
}


#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Idle constructor
 */
Texture3D::Texture3D():Texture(GL_TEXTURE_3D) {
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Create 3D texture
 *
 * @param width Width of texture
 * @param height Height of texture
 * @param depth Depth of texture (z-axis)
 * @param type Data type for the texture
 * @param channels Number of channels per voxel
 * @param clear Clear texture initially
 *
 * Creates an empty texture (allocates handle and assigns parameters) and optionally clears the
 * texture by uploading null image data.
 */
Texture3D::Texture3D(int width, int height, int depth, pixtype type, int channels, bool clear) : Texture(GL_TEXTURE_2D, channels, type),
    width_(width), height_(height), depth_(depth) {
    createHandle();
    assert(*(handle_.get()) != 0);
    glBindTexture(GL_TEXTURE_3D, *(handle_));
    updateParams();
    if (clear) this->clear();
}
#endif


#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Destructor
 *
 * Decreases reference count on the handle and deallocates texture completely when reaching 0. On
 * pooled textures, texture is released to the pool.
 */
Texture3D::~Texture3D() {
#ifdef DEBUG
    if (handle_.unique() && !fromPool_ && !handleOwned_) {
        allocTextureMemory_.fetch_sub(size());
    }
#endif
    if (fromPool_) fromPool_->releaseTexture(handle_);
}
#endif


#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Overloaded assignment operator
 *
 * @param src Object which contents should be assigned to the current ont
 *
 * @return Reference to target object after assignment
 */
Texture3D & Texture3D::operator=(const Texture3D& src) {
    if (this == &src) return *this;
#ifdef DEBUG
    if (handle_.unique() && !fromPool_ && !handleOwned_) {
        allocTextureMemory_.fetch_sub(size());
    }
#endif
    Texture::operator=(src);
    width_ = src.width_;
    height_ = src.height_;
    depth_  = src.depth_;
    memcpy(wrapMode_, src.wrapMode_, sizeof(wrapMode_));
    memcpy(interpolation_, src.interpolation_, sizeof(interpolation_));
    return *this;
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Force texture invalidation
 *
 * For non-pooled textures, this resets the (shared) handle pointer and for pooled textures, the
 * texture is released back into the pool.
 */
void Texture3D::reset() {
#ifdef DEBUG
    if (handle_.unique() && !fromPool_ && handleOwned_) {
        allocTextureMemory_.fetch_sub(size());
    }
#endif
    if (fromPool_) {
        fromPool_->releaseTexture(handle_);
        fromPool_ = nullptr;
    }
    Texture::reset();
}
#endif


#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Set texture wrap mode (u, v and w separately)
 *
 * @param uWrap Wrap mode for u-axis (also named s-axis), e.g. EDGE_CLAMP
 * @param vWrap Wrap mode for v-axis (also named t-axis), e.g. EDGE_CLAMP
 * @param wWrap Wrap mode for w-axis (also named r-axis), e.g. EDGE_CLAMP
 */
void Texture3D::wrapMode(wrap uWrap, wrap vWrap, wrap wWrap) {
    wrapMode_[0] = uWrap;
    wrapMode_[1] = vWrap;
    wrapMode_[2] = wWrap;
    paramPending_ = true;
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @copydoc Texture2D::interpolation
 */
void Texture3D::interpolation(intp minIntp, intp magIntp) {
    interpolation_[0] = minIntp;
    interpolation_[1] = magIntp;
    paramPending_ = true;
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @copydoc Texture2D::size
 */
uint32_t Texture3D::size() const {
    return width_ * height_ * depth_ * channels_ * channelSize(dataType_);
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @copydoc Texture2D::unbind
 */
void Texture3D::unbind(int unit) const {
    if (unit >= 0) glActiveTexture(GL_TEXTURE0+unit);
    glBindTexture(GL_TEXTURE_3D, 0);
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @copydoc Texture2D::bind
 */
void Texture3D::bind(int unit) const {
    if (unit >= 0) glActiveTexture(GL_TEXTURE0+unit);
    glBindTexture(GL_TEXTURE_3D, *(handle_));
    if (paramPending_) {
        updateParams();
        paramPending_ = false;
    }
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Upload data to texture or dimensionalize texture
 *
 * @param data Pointer to data to be uploaded to texture memory
 *
 * @pre Texture is bound to the \c GL_TEXTURE_3D target
 *
 * This function simply uploads the data supplied in \p data to the GL driver which is then
 * uploaded to texture memory. In case a \c nullptr is supplied to \p data, nothing will be
 * uploaded and instead an empty texture of the specified dimensions will be created.
 *
 * @warning This function assumes that the supplied data pointer has exactly the same data type
 *          as the target GL texture. If you want more fine grained control, see the overloaded
 *          functions.
 *
 * @pre Texture is bound to the \c GL_TEXTURE_3D target
 *
 * @see clear()
 */
void Texture3D::upload(const void *data) {
    assert(channels_ > 0);
    assert(channels_ <= 4);
    texinfo info = textureInfo(dataType_, channels_);
    upload(data, info.intFormat, info.format, info.dataType);
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Upload voxel data to texture
 *
 * @param data Pointer to voxel data to upload
 * @param cpuDataFmt Data format of the CPU data
 *
 * @pre Texture is bound to the \c GL_TEXTURE_3D target
 */
void Texture3D::upload(const void *data, pixtype cpuDataFmt) {
    assert(channels_ > 0);
    assert(channels_ <= 4);
    texinfo info = textureInfo(cpuDataFmt, channels_);
    upload(data, info.intFormat, info.format, info.dataType);
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Upload data to texture or dimensionalize texture
 *
 * @param data Pointer to data to be uploaded to texture memory
 * @param internal Internal representation of the texture
 * @param format Format to be uploaded
 * @param type Data type to be uploaded
 *
 * @pre Texture is bound to the \c GL_TEXTURE_3D texture target
 *
 * This function simply uploads the data supplied in \p data to the GL driver which is then
 * uploaded to texture memory. In case a \c nullptr is supplied to \p data, nothing will be
 * uploaded and instead an empty texture of the specified dimensions will be created.
 *
 * @pre Texture is bound to the \c GL_TEXTURE_3D target
 *
 * @see clear()
 */
void Texture3D::upload(const void *data, GLint internal, GLenum format, GLenum type) {
#ifdef DEBUG
    glGetError();
#endif
    if (internal != internal_) {
        glTexImage3D(GL_TEXTURE_3D, 0, internal, width_, height_, depth_, 0, format, type, data);
        internal_ = internal;
    } else {
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, width_, height_, depth_, format, type, data);
    }
#ifdef DEBUG
    GLenum err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
}
#endif


#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @copydoc Texture2D::clear
 */
void Texture3D::clear() {
    upload(nullptr);
}
#endif


/**
 * @brief Constructor to (temporarily) wrap a raw GL texture handle in a Texture2D object
 *
 * @param handle GL handle to wrap
 * @param width Width of the texture behind the handle
 * @param height Height of the texture behind the handle
 * @param type Data format
 * @param channels Number of channels
 * @param target
 */
Texture2DRef::Texture2DRef(GLuint handle, int width, int height, pixtype type,
                           int channels, GLenum target) :
    Texture2D(width,height) {
    assert(handle != 0);
    createHandle(handle, false);
    channels_ = channels;
    dataType_ = type;
    target_ = target;
    handleOwned_ = false;
    glBindTexture(GL_TEXTURE_2D, *(handle_));
    updateParams();
    // NOTE (mw) do not increase alloc count here, since this texture is not ours to track
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Retrieve per-channel size for a datatype
 *
 * @param type Data type
 *
 * @return Per-channel size (in bytes) for the supplied \p type
 */
int Texture::channelSize(pixtype type) {
    if (type == INVALID) return 0;
    if (type <= UINT8_INTEGRAL) return 1;
    if (type <= FLOAT16) return 2;
    return 4;
}


/**
 * @brief Private constructor for empty 2D texture with size set
 *
 * @param width Width of the texture
 * @param height Height of the texture
 *
 * Note that this texture is not valid, it only has the dimensions set.
 */
Texture2D::Texture2D(int width, int height) : Texture(GL_TEXTURE_2D), width_(width), height_(height) {
}


/**
 * @brief Update texture parameters in the GL state machine
 *
 * @pre Texture is bound to the \c GL_TEXTURE_2D target
 */
void Texture2D::updateParams() const {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (wrapMode_[0] == EDGE_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (wrapMode_[1] == EDGE_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (interpolation_[0] == NEAREST) ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (interpolation_[1] == NEAREST) ? GL_NEAREST : GL_LINEAR);
}

#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Idle constructor for 3D
 *
 * @param width Width of texture (voxels)
 * @param height Height of texture (voxels)
 * @param depth Depth of texture (voxels)
 */
Texture3D::Texture3D(int width, int height, int depth) : Texture(GL_TEXTURE_3D),
      width_(width), height_(height), depth_(depth) {
}
#endif



#if !defined(FYUSENET_USE_WEBGL) && !defined(FYUSENET_USE_EGL)
/**
 * @brief Update texture parameters in GL texture object
 *
 * @pre Texture is bound to \c GL_TEXTURE_3D target
 */
void Texture3D::updateParams() const {
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, (wrapMode_[0] == EDGE_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, (wrapMode_[1] == EDGE_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, (wrapMode_[2] == EDGE_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, (interpolation_[0] == NEAREST) ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, (interpolation_[1] == NEAREST) ? GL_NEAREST : GL_LINEAR);
}
#endif

/*##################################################################################################
#                   E X P L I C I T    T E M P L A T E    I N S T A N T I A T I O N                #
##################################################################################################*/

template void Texture2D::download<uint8_t>(uint8_t *target);
template void Texture2D::download<float>(float *target);

} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
