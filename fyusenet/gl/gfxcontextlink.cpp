//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context Link (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <mutex>
#include <atomic>
#include <set>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpu/gfxcontextlink.h"
#include "glcontext.h"
#include "glexception.h"
#include "../gpu/gfxcontextmanager.h"
#include "../common/miscdefs.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet {

//-------------------------------------- Local Definitions -----------------------------------------

const GfxContextLink GfxContextLink::EMPTY = GfxContextLink(true);

#ifdef DEBUG
static std::atomic<uint64_t> CONTEXT_ID_SEQCTR{1};
static std::atomic<bool> CONTEXT_ID_SPINNER{false};
static std::set<GfxContextLink *> ACTIVE_GLCTX_LINKS = std::set<GfxContextLink *>();

#define CONTEXT_ID_CRITICAL(code) { \
                                    bool expect = false; \
                                    while (!CONTEXT_ID_SPINNER.compare_exchange_strong(expect, true)) {expect=false;} \
                                    code; \
                                    CONTEXT_ID_SPINNER.store(false); \
                                  }
#endif



/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Construct context link to supplied context
 *
 * @param wrap Pointer to OpenGL context which should be linked to
 *
 * @post Link counter on the specified context is increased by 1
 */
GfxContextLink::GfxContextLink(opengl::GLContextInterface *wrap) {
    if (!wrap) {
        context_ = GfxContextManager::findCurrentContext(wrap);
    }
    else context_ = wrap;
    if (context_) {
        context_->addLink();
    }
#ifdef DEBUG
    if (context_) {
        id_ = CONTEXT_ID_SEQCTR.fetch_add(1);
        CONTEXT_ID_CRITICAL(ACTIVE_GLCTX_LINKS.insert(this))
    }
#endif
}


/**
 * @brief Remove context link to wrapped context
 *
 * @post Link counter on the wrapped context is decreased by 1
 */
GfxContextLink::~GfxContextLink() {
    if (context_) {
        context_->remLink();
#ifdef DEBUG
        CONTEXT_ID_CRITICAL(ACTIVE_GLCTX_LINKS.erase(this))
#endif
    }
    context_ = nullptr;
}


/**
 * @brief Copy link to context
 *
 * @param src Source link to copy from
 *
 * @post Link counter on wrapped context is increased by 1
 */
GfxContextLink::GfxContextLink(const GfxContextLink& src) {
    context_ = src.context_;
    if (context_) {
        context_->addLink();
#ifdef DEBUG
        id_ = CONTEXT_ID_SEQCTR.fetch_add(1);
        CONTEXT_ID_CRITICAL(ACTIVE_GLCTX_LINKS.insert(this))
#endif
    }
}


/**
 * @brief Assign link to existing object
 *
 * @param src Source link to assign to the current object
 *
 * @return Reference to current object, with new link
 *
 * @post Link counter on old (previous) context is decremented by 1, link counter of
 *       context supplied by \p src is incremented by 1
 */
GfxContextLink& GfxContextLink::operator=(const GfxContextLink& src) {
    if (this == &src) return *this;
    bool empty = false;
    if (context_) {
        context_->remLink();
    } else empty = true;
    context_ = src.context_;
    if (context_) {
        context_->addLink();
#ifdef DEBUG
        if (empty) {
            id_ = CONTEXT_ID_SEQCTR.fetch_add(1);
            CONTEXT_ID_CRITICAL(ACTIVE_GLCTX_LINKS.insert(this))
        }
#endif
    }
    return *this;
}


/**
 * @brief Issue fence sync on pipeline of linked GL Context
 *
 * @return Sync ID that was issued
 *
 * @see https://www.khronos.org/opengl/wiki/Sync_Object
 * @see waitSync
 *
 * @note Use #removeSync to delete an expired sync object to avoid leakage
 */
GfxContextLink::syncid GfxContextLink::issueSync() const {
    if (!context_) THROW_EXCEPTION_ARGS(opengl::GLException,"No context associated with link");
    assert(isCurrent());
    CLEAR_GFXERR_DEBUG
    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
#ifdef DEBUG
    int err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
    return sync;
}

/**
 * @brief Remove a sync object from context
 *
 * @param sync ID of the Sync object to remove from the GL context
 *
 * @see https://www.khronos.org/opengl/wiki/Sync_Object
 */
void GfxContextLink::removeSync(syncid sync) const {
    if (!context_) THROW_EXCEPTION_ARGS(opengl::GLException,"No context associated with link");
    assert(isCurrent());
    glDeleteSync(sync);
}


/**
 * @brief Wait for a fence sync to appear in the pipeline
 *
 * @param sync Sync ID to wait for in the pipeline
 *
 * @see https://www.khronos.org/opengl/wiki/Sync_Object
 * @see removeSync
 */
void GfxContextLink::waitSync(syncid sync) const {
    if (!context_) THROW_EXCEPTION_ARGS(opengl::GLException,"No context associated with link");
    assert(isCurrent());
    CLEAR_GFXERR_DEBUG
    glWaitSync(sync,0,GL_TIMEOUT_IGNORED);
#ifdef DEBUG
    int err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
}


/**
 * @brief Wait for a fence sync to appear in the pipeline on the client side
 *
 * @param sync Sync ID to wait for in the pipeline
 *
 * @param timeoutNS Maximum time (in nanoseconds) to wait for the provided \p sync to appear in the
 *                  pipeline
 *
 * @retval true Supplied sync object was detected on the pipeline
 * @retval false Operation timed out or there was an error
 *
 * @see https://www.khronos.org/opengl/wiki/Sync_Object
 * @see removeSync
 */
bool GfxContextLink::waitClientSync(syncid sync, GLuint64 timeoutNS) const {
    if (!context_) THROW_EXCEPTION_ARGS(opengl::GLException,"No context associated with link");
    assert(isCurrent());
    CLEAR_GFXERR_DEBUG
    GLenum rc = glClientWaitSync(sync,GL_SYNC_FLUSH_COMMANDS_BIT,timeoutNS);
#ifdef DEBUG
    int err = glGetError();
    assert(err == GL_NO_ERROR);
#endif
    if (rc == GL_TIMEOUT_EXPIRED) return false;
    if (rc == GL_WAIT_FAILED) THROW_EXCEPTION_ARGS(opengl::GLException,"Error while waiting for GL sync");
    if (rc == GL_CONDITION_SATISFIED || rc == GL_ALREADY_SIGNALED) return true;
    return false;
}


/**
 * @brief Obtain pointer to texture pool usable with the context
 *
 * @return Pointer to texture pool or \c nullptr if no pool exists
 */
opengl::ScopedTexturePool * GfxContextLink::texturePool() const {
    return interface()->texturePool();
}


/**
 * @brief Check if linked context is valid and current to this thread
 *
 * @retval true Context is valid and is current to this thread
 * @retval false Context is either not valid or not current to this thread
 *
 * OpenGL contexts can only be bound to one thread at a time. This function checks if the
 * context linked to by this object is valid and if is the context that is attached to the
 * current thread.
 */
bool GfxContextLink::isCurrent() const {
    if (!context_) return false;
    return context_->isCurrent();
}


/**
 * @brief Invalidate the link (not the context)
 *
 * @post Link count on the wrapped context is decremented by 1
 */
void GfxContextLink::reset() {
    if (context_) {
        context_->remLink();
#ifdef DEBUG
        CONTEXT_ID_CRITICAL(ACTIVE_GLCTX_LINKS.erase(this))
#endif
    }
    context_ = nullptr;
}


/**
 * @brief Obtain device number for multi-GPU systems
 *
 * @return Device ID or -1 if context is not valid
 *
 * For multi-GPU systems (which are not fully supported yet), this returns the ID of the GPU/device
 * that is hosting the context.
 */
int GfxContextLink::device() const {
    if (!context_) return -1;
    return context_->device();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor to generate empty links
 *
 * @param empty Dummy
 */
GfxContextLink::GfxContextLink(bool empty) : context_(nullptr) {
}

} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
