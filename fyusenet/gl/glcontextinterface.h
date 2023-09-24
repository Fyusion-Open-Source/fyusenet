//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context Abstraction Interface (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

#include <cassert>
#include <atomic>

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {

namespace fyusenet {
    class GfxContextLink;
}

namespace opengl {

class PBOPool;
class ScopedTexturePool;

/**
 * @brief Interface class for a slightly abstracted GL context
 *
 * This interface exposes essential functionality of GL contexts while (slightly) abstracting from
 * the underlying GL platform (e.g. desktop GL, EGL, WebGL) and operating system.
 *
 * The actual context wrapper (GLContext) derives from this interface class and has slightly more
 * system-specific functionality that should not be exposed outside the internal workings of the
 * GL abstraction layer (if possible).
 *
 * @see GfxContextManager, GLContext
 */
class GLContextInterface {
    friend class fyusenet::GfxContextLink;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    /**
     * @brief Idle constructor
     *
     * @param idx Context index (see GfxContextManager)
     * @param dev Device ID/index the context runs on
     */
    GLContextInterface(int idx, int dev) : links_(0), index_(idx), deviceID_(dev) {
    }

    /**
     * @brief Idle destructor
     */
    virtual ~GLContextInterface() {
    }

    // ------------------------------------------------------------------------
    // Interface methods
    // ------------------------------------------------------------------------

    /**
     * @brief Check if a context is (potentially) a derived context
     *
     * @retval true if context was derived from a main context
     * @retval false if context is the main context
     */
    bool isDerived() const {
        return (getMain() != this);
    }

    /**
     * @brief Retrieve pointer to PBOPool that pools PBOs for writing/upload purposes
     *
     * @return Pointer to %PBO pool or \c nullptr if no such pool exists
     */
    virtual PBOPool * getWritePBOPool() const = 0;

    /**
     * @brief getReadPBOPool
     * @return
     */
    virtual PBOPool * getReadPBOPool() const = 0;


    /**
     * @brief Make GL context current to the calling thread
     *
     * @retval true if context was made the current one
     * @retval false otherwise
     */
    virtual bool makeCurrent() const = 0;


    /**
     * @brief Obtain pointer to texture pool that is valid for this context
     *
     * @return Pointer to texture pool or \c nullptr if there was no texture pool allocated
     */
    virtual ScopedTexturePool * texturePool() const = 0;


    /**
     * @brief Release current GL context from the calling thread
     *
     * @retval true if context was released
     * @retval false otherwise
     */
    virtual bool releaseCurrent() const = 0;


    /**
     * @brief Initialize GL context
     */
    virtual void init() = 0;

    /**
     * @brief Synchronize GPU pipeline / flush-out pending commands
     */
    virtual void sync() const = 0;

    /**
     * @brief Check if context is current to the calling thread
     *
     * @retval true if GL context is current to the calling thread
     * @retval false otherwise
     */
    virtual bool isCurrent() const = 0;

    /**
     * @brief Make context use default system surface
     *
     * GL contexts can be attached to different surfaces. This function instructs the context to
     * use the default surface.
     *
     * @note This concept is not necessarily existing on all platforms
     */
    virtual void useDefaultSurface() = 0;

    /**
     * @brief Checks if context is derived/shared with a main context
     *
     * @param main Main context to check for sharing
     *
     * @retval true if this context was derived from the supplied \p main context
     * @retval false otherwise
     *
     * For sake of simplicity we assume that a group of shared contexts are all shared with the
     * same main context instead of for example being daisy-chained. This function checks if
     * this context was derived from / is shared with the supplied \p main context.
     */
    virtual bool isDerivedFrom(const GLContextInterface * main) const = 0;

    /**
     * @brief Retrieve pointer to main GL context interface
     *
     * @return Main GL context interface
     *
     * In case of derived contexts, this function will return the main context interface. If the
     * context itself is the main context, it will return a pointer to itself.
     */
    virtual GLContextInterface * getMain() const = 0;

    /**
     * @brief Compute 64-bit hash for this context
     *
     * @return Hash value for this context (hopefully unique :-) )
     */
    virtual uint64_t hash() const = 0;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Get device ID this context was created on
     *
     * @return Device ID (e.g. GPU index)
     */
    int device() const {
        return deviceID_;
    }

    /**
     * @brief Retrieve index of this context
     *
     * @return Index as determined by GfxContextManager
     *
     * All contexts are managed by the GfxContextManager which assigned indices to context. This
     * function returns the index assigned by the GfxContextManager.
     */
    int index() const {
        return index_;
    }

    /**
     * @brief Get derived index for context
     *
     * @return Derived index of this context or -1 if this context was not derived from another
     *         context
     *
     * Shared GL contexts are implemented by "deriving" from a main context. A derived context is
     * simply a context that was created to be shared with a given main context. As a main context
     * can have more than one derived context, it comes in handy to assign "derived indices" for the
     * derived contexts. A derived index always pertains to the same main context.
     */
    int derivedIndex() const {
        return derivedIdx_;
    }


    /**
     * @brief Clear framebuffer to specific color
     *
     * @param red Red color component to clear framebuffer to
     * @param green Green color component to clear framebuffer to
     * @param blue Blue color component to clear framebuffer to
     * @param alpha Alpha component to clear framebuffer to
     */
    void clear(float red=0.0f, float green=0.0f, float blue=0.0f, float alpha=0.0f) {
        glClearColor(red, green, blue, alpha);
        glClear(GL_COLOR_BUFFER_BIT);
    }


    /**
     * @brief Get usage/link counter for this context
     *
     * @return Number of links to this context
     */
    int uses() const {
        return links_.load();
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Add a link to this context
     *
     * This increases the active link counter of this context by 1
     */
    void addLink() {
        links_.fetch_add(1);
    }

    /**
     * @brief Remove a link from this context
     *
     * This decreases the active link counter of this context by 1
     */
    void remLink() {
        links_.fetch_add(-1);
        assert(links_.load() >= 0);
    }

    std::atomic<int> links_;            //!< Number of active links to this context
    int index_ = 0;                     //!< Index of this context in a globally managed context list (see GfxContextManager)
    int derivedIdx_ = -1;               //!< For derived (=shared) contexts, the index of the context within a derived list
    int deviceID_ = 0;                  //!< Device ID (e.g. GPU index) that this context runs on
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
