//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Texture Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

class BasicTexturePool;

/**
 * @brief Base class for OpenGL textures
 *
 * @note We are not tracking GL contexts in textures because we assume context sharing for those.
 */
class Texture {
    friend class BasicTexturePool;
 public:

    /**
     * @brief Enumerator for texture clamp modes
     */
    enum wrap : uint8_t {
        EDGE_CLAMP = 0,             //!< Clamp to edge
        REPEAT                      //!< Texture repeat
    };

    /**
     * @brief Enumerator for texture interpolation modes
     */
    enum intp : uint8_t {
        NEAREST = 0,                //!< Nearest neighbor interpolation
        LINEAR                      //!< Linear interpolation
    };

    /**
     * @brief Enumerator for data types
     */
    enum pixtype : uint8_t {
        INVALID = 0,                //!< Unsupported/invalid datatype
        UINT8,                      //!< Unsigned 8-bit (normalized)
        UINT8_INTEGRAL,             //!< Unsigned 8-bit (integer)
        UINT16,                     //!< Unsigned 16-bit (normalized)
        UINT16_INTEGRAL,            //!< Unsigned 16-bit (integer)
        INT16_INTEGRAL,             //!< Signed 16-bit (integer)
        FLOAT16,                    //!< Half-precision floating-point (16-bit)
        FLOAT32                     //!< Single-precision floating-point (32-bit)
    };

    /**
     * @brief Compound structure that encapsulates basic texture-typing information
     *
     * OpenGL uses more than one parameter for "typing" a texture. These include:
     *  - Internal texture format, a.k.a. sized texture format
     *  - Texture format (a.k.a. unsized texture format)
     *  - Data type
     *
     *  The \e internal texture format specifies the number of channels per pixel, as well as the
     *  number of bits per channel. For example \c GL_RGBA8 defines a 4-channel RGBA texture where
     *  each channel stores 8-bit (integer) information. The (non-internal) texture format provides
     *  the number of channels only, e.g. \c GL_RGBA to define a 4-channel texture. Finally, the
     *  data type defines what type of fundamental data type is used, e.g. \c GL_UNSIGNED_BYTE
     *  would indicate an 8-bit unsigned value per channel.
     *
     *  This information in conveniently stored in this structure, together with our own data-type,
     *  which differentiates between integral and non-integral textures on the data-type.
     *
     * @see https://www.khronos.org/opengl/wiki/Texture
     */
    struct texinfo {
        texinfo(GLint intf, GLenum fmt, GLenum dtype, opengl::Texture::pixtype ptype) : intFormat(intf), format(fmt), pixType(ptype), dataType(dtype) {}
        GLint intFormat = 0;
        GLenum format = 0;
        pixtype pixType = opengl::Texture::pixtype::INVALID;
        GLenum dataType = GL_UNSIGNED_BYTE;
    };

    // ------------------------------------------------------------------------
    // Constructors / Destructor
    // ------------------------------------------------------------------------
    /**
     * @brief Constructor (idle)
     */
    Texture() {}

    /**
     * @brief Constructor
     *
     * @param tgt Texture target (e.g. \c GL_TEXTURE_2D for 2D textures)
     * @param channels Number of channels for the texture
     * @param type Data type for the texture
     *
     */
    Texture(GLenum tgt, uint8_t channels=0, pixtype type = INVALID) :
          target_(tgt), channels_(channels), dataType_(type) {}

    /**
     * @brief Destructor (idle)
     *
     * See the derived classes destructors for action.
     */
    virtual ~Texture() {
    }

    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    Texture& operator=(const Texture& src);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    bool isIntegral() const;
    bool isFloat() const;
    void setTarget(GLenum target);

    static texinfo textureInfo(pixtype type, int channels);

    /**
     * @brief Get raw GL handle wrapped by the texture
     *
     * @return Raw GL handle
     */
    GLuint getHandle() const{
        if (!handle_.get()) return 0;
        return *handle_.get();
    }

    /**
     * @brief Check if texture is empty (has no raw GL handle)
     *
     * @retval true if texture is empty
     * @retval false otherwise
     */
    bool empty() const {
        return (getHandle() == 0);
    }

    /**
     * @brief Force texture invalidation
     *
     * Also see implementation in derived classes
     */
    void reset() {
        handle_.reset();
    }

    /**
     * @brief Get (default) texture target
     *
     * @return Target for this type of texture (e.g. \c GL_TEXTURE_2D for 2D textures)
     */
    GLenum target() const {
        return target_;
    }


    /**
     * @brief Get data type for texture
     *
     * @return Texture data type
     */
    pixtype type() const {
        return dataType_;
    }

    /**
     * @brief Get number of channels for the texture
     *
     * @return Number of channels
     */
    uint8_t channels() const {
        return channels_;
    }


    /**
     * @brief unique
     *
     * @retval true
     * @retval false
     */
    bool unique() const {
        if (fromPool_) return (handle_.use_count() == 2);
        else return handle_.unique();
    }

    /**
     * @brief Get texture reference count
     *
     * @return Number of times the underlying handle is referenced
     */
    int refcount() const {
        return handle_.use_count();
    }

    /**
     * @brief Get sync ID stored in a (fenced) texture
     *
     * @return Synchronization ID or 0 if no sync ID was set
     *
     * @see https://www.khronos.org/opengl/wiki/Sync_Object
     * @see GfxContextLink::issueSync, GfxContextLink::removeSync, GfxContextLink::waitSync
     *
     * @warning This function is not thread-safe (yet)
     */
    GLsync syncID() const {
        return syncID_;
    }

    /**
     * @brief Query fence state variable
     *
     * @retval true if fence is desired
     * @retval false otherwise
     *
     * Textures that are shared between different contexts might not be in sync. Fencing is way
     * to make sure that textures are in sync. This function queries the texture for the fence
     * flag being set via the #fence call.
     *
     * @see https://www.khronos.org/opengl/wiki/Sync_Object
     * @see syncID()
     * @see GfxContextLink::issueSync, GfxContextLink::removeSync, GfxContextLink::waitSync
     *
     * @warning This function is not thread-safe (yet)
     */
    bool wantsFence() const {
        return wantsFence_;
    }

    /**
     * @brief Set the fencing state for the texture and add an optional sync ID
     *
     * @param id Optional synchronization ID which was obtained by GfxContextLink::issueSync
     *
     * This function marks the texture to require fencing in order to make sure that the
     * contents are available at read time. Tracking of the corresponding sync ID can either be done
     * completely externally, or the texture object may be used to store the sync ID.
     *
     * @see https://www.khronos.org/opengl/wiki/Sync_Object
     * @see GfxContextLink::issueSync, GfxContextLink::removeSync, GfxContextLink::waitSync
     *
     * @warning This function is not thread-safe (yet)
     */
    void fence(GLsync id = 0) {
        wantsFence_ = true;
        syncID_ = id;
    }

    static int64_t usedTextureMemory();
    static int channelSize(pixtype type);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Create raw GL texture handle
     */
    void createHandle() {
        GLuint handle = 0;
        glGenTextures(1, &handle);
        assert(handle != 0);
        createHandle(handle, true);
    }

    /**
     * @brief Create shared handle pointer to raw GL texture handle
     *
     * @param handle Handle to create shared pointer to
     *
     * @param owned if \c true, the handle is owned by this instance and should be deleted from
     *              the GL texture objects once the usage count reaches 0
     */
    void createHandle(GLuint handle, bool owned) {
        if (owned) handle_ = std::shared_ptr<GLuint>(new GLuint[1], deleteOwnedHandle);
        else handle_ = std::shared_ptr<GLuint>(new GLuint[1], deleteExternalHandle);
        *(handle_.get()) = handle;
        assert(*(handle_.get()) == handle);
    }


    /**
     * @brief Deletection function for owned handles
     *
     * @param handlePtr Pointer to handle
     *
     * This removes the GL texture object referenced to by the handle and also deallocates the
     * memory occupied by the handle.
     */
    static void deleteOwnedHandle(GLuint * handlePtr) {
        glDeleteTextures(1, handlePtr);
        delete [] handlePtr;
    }


    /**
     * @brief Deletection function for owned handles
     *
     * @param handlePtr Pointer to handle
     *
     * This only dedeallocates the memory occupied by the handle, the referenced texture will not
     * be deleted.
     */
    static void deleteExternalHandle(GLuint * handlePtr) {
        delete [] handlePtr;
    }

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::shared_ptr<GLuint> handle_;                      //!< Shared pointer to raw GL handle
    bool handleOwned_ = true;                             //!< Indicator if texture handle is owned by this class or just externally tracked
    mutable bool paramPending_ = false;                   //!< Indicator that texture parameters have changed in the class but not (yet) in the GL object
    BasicTexturePool * fromPool_ = nullptr;               //!< Pointer to texture pool if this is a pooled texture
    GLenum target_ = 0;                                   //!< Texture target for this texture (e.g. \c GL_TEXTURE_2D )
    uint8_t channels_ = 0;                                //!< Number of channels per pixel
    pixtype dataType_ = INVALID;                          //!< Data type for texture
    GLint internal_ = 0;                                  //!< OpenGL internal format (a.k.a. sized format) for this texture
    bool wantsFence_ = false;                             //!< Indicator that the re-use of the texture requires a fence operation for safe re-use
    GLsync syncID_ = 0;                                   //!< Synchronization ID for fenced textures (optional, defaults to 0)
    static std::atomic<int64_t> allocTextureMemory_;      //!< For debug & statistics, tracking counter of allocated texture memory
};


/**
 * @brief Simple wrapper for 2D textures
 */
class Texture2D : public Texture {
 public:
    // ------------------------------------------------------------------------
    // Constructors / Destructor
    // ------------------------------------------------------------------------
    Texture2D();
    Texture2D(int width, int height, pixtype type, int channels, BasicTexturePool *pool, bool lock=true);
    Texture2D(int width, int height, pixtype type, int channels, bool clear=false);
    virtual ~Texture2D();
    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    Texture2D& operator=(const Texture2D& src);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void wrapMode(wrap uWrap, wrap vWrap);
    void interpolation(intp minIntp, intp magIntp);
    void bind(int unit=0) const;
    void unbind(int unit=0) const;
    void upload(const void *data);
    void upload(const void *data, pixtype cpuDataFmt);
    void upload(const void *data, GLint internal, GLenum format, GLenum type);
    void clear();
    void reset();

    /**
     * @brief Get texture width
     *
     * @return Width (in pixels)
     */
    int width() const {
        return width_;
    }

    /**
     * @brief Get texture height
     *
     * @return Height (in pixels)
     */
    int height() const {
        return height_;
    }

    uint32_t size() const;

    // ------------------------------------------------------------------------
    // Testing functions (unit-testing)
    // ------------------------------------------------------------------------
    template<typename T>
    void download(T *target);

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    Texture2D(int width, int height);
    void updateParams() const;

    // ------------------------------------------------------------------------
    // Member Variables
    // ------------------------------------------------------------------------
    int width_ = 0;                                     //!< Texture width (pixels)
    int height_ = 0;                                    //!< Texture height (pixels)
    wrap wrapMode_[2] = {EDGE_CLAMP, EDGE_CLAMP};       //!< u,v wrap modes for the texture
    intp interpolation_[2] = {NEAREST, NEAREST};        //!< Interpolation modes for the texture
};


/**
 * @brief Simple wrapper for 3D textures
 */
class Texture3D : public Texture {
 public:
    // ------------------------------------------------------------------------
    // Constructors / Destructor
    // ------------------------------------------------------------------------
    Texture3D();
    Texture3D(int width, int height, int depth, pixtype type, int channels, bool clear=false);
    virtual ~Texture3D();

    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    Texture3D& operator=(const Texture3D& src);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void wrapMode(wrap uWrap, wrap vWrap, wrap wWarp);
    void interpolation(intp minIntp, intp magIntp);
    void bind(int unit=0) const;
    void unbind(int unit=0) const;
    void upload(const void *data);
    void upload(const void *data, pixtype cpuDataFmt);
    void upload(const void *data, GLint internal, GLenum format, GLenum type);
    void clear();
    void reset();

    /**
     * @brief Get texture width
     *
     * @return Width (in voxels)
     */
    int width() const {
        return width_;
    }

    /**
     * @brief Get texture height
     *
     * @return Height (in voxels)
     */
    int height() const {
        return height_;
    }

    /**
     * @brief Get depth (z-extent) of texture
     *
     * @return Depth (in voxels)
     */
    int depth() const {
        return depth_;
    }

    uint32_t size() const;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    Texture3D(int width, int height, int depth);
    void updateParams() const;

    // ------------------------------------------------------------------------
    // Member Variables
    // ------------------------------------------------------------------------
    int width_ = 0;             //!< Volume extent x-direction (width)
    int height_ = 0;            //!< Volume extent y-direction (height)
    int depth_ = 0;             //!< Volume extent z-direction (depth)
    /**
     * @brief u,v wrap modes for the texture
     */
    wrap wrapMode_[3] = {EDGE_CLAMP, EDGE_CLAMP, EDGE_CLAMP};
    /**
     * @brief Interpolation modes for the texture*
     */
    intp interpolation_[2] = {NEAREST, NEAREST};
};


/**
 * @brief Simple wrapper for 2D textures with pre-existing handle
 *
 * @note The ownership over the texture handle is not taken by this class
 */
class Texture2DRef : public Texture2D {
 public:
    Texture2DRef(GLuint handle, int width, int height, pixtype type, int channels, GLenum target = GL_TEXTURE_2D);
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
