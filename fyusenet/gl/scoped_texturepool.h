//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture Pool (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <atomic>

//-------------------------------------- Project  Headers ------------------------------------------

#include "texture.h"
#include "../gpu/gfxcontextlink.h"
#include "../gpu/gfxcontexttracker.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion::opengl {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Scoped OpenGL texture pool
 *
 * This class implements a texture pool where textures are kept by their dimension, number of
 * channels and pixel type. For many occasions, textures do not need to be unique and may be
 * reused. In fact, the way that textures are used in FyuseNet (mere buffers, often transient),
 * we want some kind of "oversubscribed" pool and this class seeks to facilitate that - admittedly
 * not very clean - approach.
 *
 * For oversubscribing to textures, his pool uses "scopes" in the form of simple integer IDs as tag
 * to the textures. Once a pool texture is used, the scope ID is added as a tag to that texture,
 * such that each texture is used within the same scope only once. For code modules that can make
 * even more use of texture re-use, simply use more than one scope ID. Scope IDs have to be obtained
 * from the texture pool via the scopeID() function.
 *
 * In addition to the scoping, this pool uses the concept of \e locking textures in order to
 * indicate that a texture is to be used exclusively. Textures that are locked \b must be released
 * before they are put back into the pool.
 *
 * Pools are created with a target GL context, which is the one to be used for creation of the
 * pool and destruction of the pool. As a strict enforcement of the original pool context being
 * the only usable one is too limiting for multi-threaded scenarios, the texture pool
 * <b>does not check</b> if the original context is the current one when obtaining a (new) texture.
 * In these cases it is silently assumed that if it is not the original context that is being
 * bound, it is at least a context that is \b shared with the original context.
 *
 * @see https://www.khronos.org/opengl/wiki/OpenGL_Context
 */
class ScopedTexturePool : public fyusenet::GfxContextTracker {
    /**
     * @brief Compound structure to index textures with
     */
    struct key {
        key(int w, int h, int c, Texture::pixtype t) : width(w), height(h), channels(c), type(t) {}

        bool operator==(const key& other) const {
            return ((type == other.type) &&
                    (width == other.width) &&
                    (height == other.height) &&
                    (channels == other.channels));
        }

        friend bool operator<(const key& op1, const key& op2) {
            if ((int)op1.type < (int)op2.type) return true;
            else if ((int)op1.type == (int)op2.type) {
                if (op1.channels < op2.channels) return true;
                else if (op1.channels == op2.channels) {
                    if (op1.width < op2.width) return true;
                    else if (op1.width == op2.width) {
                        return (op1.height < op2.height);
                    }
                }
            }
            return false;
        }

        int width;
        int height;
        int channels;
        Texture::pixtype type;
    };

    struct texvalue {
        texvalue(std::shared_ptr<GLuint> h, uint32_t s) : handle(std::move(h)) {
            scopes.insert(s);
        }
        std::shared_ptr<GLuint> handle;
        std::unordered_set<uint32_t> scopes;
    };

 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit ScopedTexturePool(const fyusenet::GfxContextLink & link = fyusenet::GfxContextLink());
    ~ScopedTexturePool() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setContext(const fyusenet::GfxContextLink & link) override;

    std::shared_ptr<GLuint> obtainTexture(int width, int height, int channels, Texture::pixtype type, uint32_t scope, bool lock=true);
    void unlockTexture(const Texture& tex);
    void releaseTexture(const std::shared_ptr<GLuint> & handle);
    void releaseTexture(const Texture& tex);
    [[nodiscard]] bool isLocked(GLuint handle) const;
    void garbageCollection();

    static int64_t poolMemory() {
        return allocPoolMemory_.load();
    }

    uint32_t scopeID() const {
        return scopeID_.fetch_add(1);
    }

    void logAllocationInfo();

 protected :
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    std::shared_ptr<GLuint> findTexture(int width, int height, int channels, Texture::pixtype type, uint32_t scope, bool lock);
    static void textureDel(GLuint * handlePtr);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::recursive_mutex lock_;                              //!< Locking facility for misc multi-threading sync
    std::unordered_set<GLuint> lockedTextures_;              //!< Set of texture handles that are locked
    std::multimap<key, texvalue> textures_;                  //!< Actual texture pool that maps texture sizes/types to texture handles
    static std::atomic<int64_t> allocPoolMemory_;            //!< Tracker that keeps track of allocated texture memory (for all pools)
    uint64_t hits_ = 0;                                      //!< Hit counter for the pool (how many times a texture was available that matches a query)
    uint64_t misses_ = 0;                                    //!< Miss counter for the pool (how many times a new texture had to be created to match a query)
    bool valid_ = false;                                     //!< Validity indicator
    mutable std::atomic<uint32_t> scopeID_{1};               //!< Scope identifier counter
};

} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
