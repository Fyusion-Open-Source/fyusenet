//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture Pool
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdio>
#include <cassert>
#include <cinttypes>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "basic_texturepool.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------

std::atomic<int64_t> BasicTexturePool::allocPoolMemory_;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param link Link to GL context to use for the texture pool
 *
 * Creates an empty (and valid) texture pool
 */
BasicTexturePool::BasicTexturePool(const fyusenet::GfxContextLink & link) : GfxContextTracker() {
    if (link.isValid()) {
        context_ = link;
        valid_ = true;
    }
}


/**
 * @brief Destructor
 *
 * Releases all (non-used) textures in the pool.
 *
 * @pre No textures from this pool shall be held by any other object.
 *
 * @note When some textures are still held by other instances when calling this,
 *       the allocation tracker update will not be correct.
 */
BasicTexturePool::~BasicTexturePool() {
    // NOTE (mw) when some textures are still held by other instances when calling this,
    // the allocation tracker update will not be correct.
    garbageCollection();
    valid_ = false;
}


/**
 * @brief Set (a new) GL context for the texture pool
 *
 * @param link New GL context link
 *
 * Deallocates any (non-externally held) textures from the pool and clears it to be used with
 * the new context.
 *
 * @pre The \b old context (or none) is current to the calling thread
 *
 * @note When some textures are still held by other instances when calling this,
 *       the allocation tracker update will not be correct.
 */
void BasicTexturePool::setContext(const fyusenet::GfxContextLink &link) {
    // TODO (mw) this is not really thread-safe
    if (context_.isValid()) {
        assertContext();
        garbageCollection();
        textures_.clear();
    }
    GfxContextTracker::setContext(link);
}


/**
 * @brief Obtain (and optionally lock) a texture from the texture pool
 *
 * @param width Width of the texture
 * @param height Height of the texture
 * @param channels Number of channels per pixel (1..4)
 * @param type Pixel type for the texture
 * @param lock Flag that controls whether the texture should be locked (which is the default)
 *
 * @return Shared pointer to texture handle which may be used in Texture objects
 *
 * @note This function may be called with GL contexts current that are \b not the context for which
 *       this pool was once created. In that case, the currently active context \b must be shared
 *       with the initial context.
 */
std::shared_ptr<GLuint> BasicTexturePool::obtainTexture(int width, int height, int channels, Texture::pixtype type, bool lock) {
    std::unique_lock<std::recursive_mutex> lck(lock_);
    std::shared_ptr<GLuint> result = findTexture(width, height, channels, type);
    if (!result) {
        key k(width, height, channels, type);
        GLuint handle=0;
        glGenTextures(1, &handle);
        glBindTexture(GL_TEXTURE_2D, handle);
        Texture::texinfo info = Texture::textureInfo(type, channels);
        glTexImage2D(GL_TEXTURE_2D, 0, info.intFormat, width, height, 0, info.format, info.dataType, nullptr);
#ifdef DEBUG
        allocPoolMemory_.fetch_add(width * height * channels * Texture::channelSize(type));
#endif
        result = std::shared_ptr<GLuint>(new GLuint[1], textureDel);
        *(result.get()) = handle;
        textures_.insert(std::pair<key, std::shared_ptr<GLuint>>(k, result));
        misses_++;
    } else {
        hits_++;
    }
    if (lock) lockedTextures_.insert(*result.get());
    return result;
}

/**
 * @brief Unlock a locked texture in the pool (without releasing it)
 *
 * @param tex Texture that should be unlocked
 *
 * This unlocks a texture, which allows the pool to use this texture more than once though it is
 * not being released back into the pool.
 */
void BasicTexturePool::unlockTexture(const Texture& tex) {
    std::unique_lock<std::recursive_mutex> lck(lock_);
    assert(valid_);
    lockedTextures_.erase(*(tex.handle_.get()));
}



/**
 * @brief Release texture back into the pool
 *
 * @param handle Shared pointer to texture handle that shall be released back into the pool
 *
 * Releases a texture back into the pool and also unlocks it, such that it can be re-used.
 */
void BasicTexturePool::releaseTexture(const std::shared_ptr<GLuint> & handle) {
    std::unique_lock<std::recursive_mutex> lck(lock_);
    assert(valid_);
    assert(handle.use_count() > 1);
    if (handle.use_count() == 2) {                  // NOTE (mw) not super happy about this construct due to potential race conditions (could not construct one in error-free handling, but I don't like it)
        lockedTextures_.erase(*(handle.get()));
    }
}


/**
 * @brief Release texture back into the pool
 *
 * @param tex Texture that should be released (and unlocked)
 *
 * Overloaded convenience function.
 */
void BasicTexturePool::releaseTexture(const Texture& tex) {
    releaseTexture(tex.handle_);
}


/**
 * @brief Check if a texture handle is in the pool and locked
 *
 * @param handle GL texture handle to check
 *
 * @retval true if texture is locked
 * @retval false if texture is not locked
 */
bool BasicTexturePool::isLocked(GLuint handle) const {
    assert(valid_);
    return (lockedTextures_.find(handle) != lockedTextures_.end());
}


/**
 * @brief Perform garbage collection of textures that are currently unused
 *
 * @pre This function must be called with the original GL context that the pool was created with
 *      being the active one to the calling thread.
 *
 * @throws Error if called with the wrong context
 */
void BasicTexturePool::garbageCollection() {
    std::unique_lock<std::recursive_mutex> lck(lock_);
    assert(valid_);
    assertContext();
    auto ti = textures_.begin();
    while (ti != textures_.end()) {
        if (ti->second.unique()) {
#ifdef DEBUG
            {
                uint32_t extent = ti->first.width * ti->first.height;
                allocPoolMemory_.fetch_sub(extent * ti->first.channels * Texture::channelSize(ti->first.type));
            }
#endif
            lockedTextures_.erase(*ti->second.get());
            ti = textures_.erase(ti);
        } else {
            ++ti;
        }
    }
}

/**
 * @brief Debug helper that logs all allocated textures (with basic parameters) to the logging facility
 */
void BasicTexturePool::logAllocationInfo() {
#ifdef DEBUG
    std::unique_lock<std::recursive_mutex> lck(lock_);
    for (auto ti = textures_.begin(); ti != textures_.end(); ++ti) {
        FNLOGD("Pool texture %d:\n", *ti->second.get());
        FNLOGD("  width: %d\n  height: %d  channels: %d\n  type: %d\n",ti->first.width, ti->first.height, ti->first.channels, (int)ti->first.type);
    }
    FNLOGD("Hits: %" PRIu64, hits_);
    FNLOGD("Misses: %" PRIu64, misses_);
#endif
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Check if a texture that meets a set of query criteria is in the pool
 *
 * @param width Width of texture (in pixels)
 * @param height Height of texture (in pixels)
 * @param channels Number of channels per pixel
 * @param type Data type
 *
 * @return Shared pointer to texture handle in case it was found in the pool, empty if not found.
 *
 * This function checks if a texture that matches the query criteria is in the pool and is not locked. If
 * such a texture is found, a shared pointer to the texture handle is returned. An empty shared pointer is
 * returned if no such texture was found.
 *
 * @see Texture::pixtype
 */
std::shared_ptr<GLuint> BasicTexturePool::findTexture(int width, int height, int channels, Texture::pixtype type) {
    std::unique_lock<std::recursive_mutex> lck(lock_);
    assert(valid_);
    assert(width > 0);
    assert(height > 0);
    assert(channels > 0);
    key k(width, height, channels, type);
    auto ii = textures_.find(k);
    if (ii != textures_.end()) {
        while (ii != textures_.end() && ii->first == k) {
            GLuint hdl = *(ii->second.get());
            if (lockedTextures_.find(hdl) == lockedTextures_.end()) return ii->second;
            else ++ii;
        }
    }
    return std::shared_ptr<GLuint>();
}

/**
 * @brief Texture handle deallocator for the shared pointer
 *
 * @param handlePtr Pointer to texture handle that should be deleted
 *
 * Deallocator that is used in the shared pointers created by this pool's texture allocator
 */
void BasicTexturePool::textureDel(GLuint * handlePtr) {
    // TODO (mw) check for context ?
    if (handlePtr) {
        glDeleteTextures(1, handlePtr);
        delete [] handlePtr;
    }
}


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
