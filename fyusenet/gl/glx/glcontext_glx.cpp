//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context for GLX
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#if defined(FYUSENET_USE_EGL) || defined(__APPLE__) || !defined(__linux__)
#error THIS FILE SHOULD NOT BE COMPILED
#else

//--------------------------------------- System Headers -------------------------------------------

#include <atomic>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../common/logging.h"
#include "../glexception.h"
#include "../glcontext.h"
#include "../../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------



namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------


typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);

#define GLV_MAJOR 4
#define GLV_MINOR 3

int context_attribs[] = {
  GLX_CONTEXT_MAJOR_VERSION_ARB ,GLV_MAJOR,
  GLX_CONTEXT_MINOR_VERSION_ARB, GLV_MINOR,
  GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
  GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
  None
};


int visualAttribs[] = {None};

int pBufferAttribs[] = {
  GLX_PBUFFER_WIDTH,  GLContext::DEFAULT_WIDTH,
  GLX_PBUFFER_HEIGHT, GLContext::DEFAULT_HEIGHT,
  None
};

glXCreateContextAttribsARBProc glXCreateContextAttribsARB = nullptr;
glXMakeContextCurrentARBProc glXMakeContextCurrentARB   = nullptr;

static std::atomic<bool> THREAD_INIT{false};

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Idle constructor
 *
 * @param idx Context index as given by the GfxContextManager
 * @param device Device ID to run the context on
 * @param manager Pointer to GfxContextManager instance that manages this context
 * @param width Width of underlying surface
 * @param height Height of underlying surface
 */
GLContext::GLContext(int idx, int device, fyusenet::GfxContextManager * manager, int width, int height) :
    GLContextInterface(idx, device), width_(width), height_(height), manager_(manager) {
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
    glXMakeContextCurrentARB   = (glXMakeContextCurrentARBProc)glXGetProcAddressARB((const GLubyte *)"glXMakeContextCurrent");
    pBufferAttribs[1] = width_;
    pBufferAttribs[3] = height_;
}


/**
 * @brief Destructor
 *
 * Takes down GL context and releases resources held by it, if context is not an internal one.
 */
GLContext::~GLContext() {
#ifdef DEBUG
    if ((uses() > 0) && (!external_)) {
        FNLOGW("Destroying GL context with %d active links, check your code", uses());
    }
#endif
    if (!external_) {
        glXMakeContextCurrentARB(displayPtr_, None, None, nullptr);
        glXDestroyContext(displayPtr_, context_);
    }
    context_ = nullptr;
    displayPtr_ = nullptr;
}



/**
 * @copydoc GLContextInterface::isCurrent()
 */
bool GLContext::isCurrent() const {
    if (!context_) return false;
    GLXContext ctx = glXGetCurrentContext();
    if (!ctx) return false;
    return (ctx == context_);
}


/**
 * @copydoc GLContextInterface::init()
 */
void GLContext::init() {
    if ((!glXCreateContextAttribsARB)||(!glXMakeContextCurrentARB)) THROW_EXCEPTION_ARGS(GLException,"Cannot lookup GLX functions for context creation");

    displayPtr_ = XOpenDisplay(nullptr);
    if (!displayPtr_) THROW_EXCEPTION_ARGS(GLException,"Cannot obtain X11 display");
    int numconfigs=0;
    GLXFBConfig *fbconfigs = glXChooseFBConfig(displayPtr_, DefaultScreen(displayPtr_), visualAttribs, &numconfigs);
    if (numconfigs<=0) THROW_EXCEPTION_ARGS(GLException,"Desired configuration not available");

    context_ = glXCreateContextAttribsARB(displayPtr_, fbconfigs[0], 0, True, context_attribs);
    pBuffer_ = glXCreatePbuffer(displayPtr_, fbconfigs[0], pBufferAttribs);
    XFree(fbconfigs);
    XSync(displayPtr_, False);
}


/**
 * @copydoc GLContextInterface::makeCurrent()
 */
bool GLContext::makeCurrent() const {
    return glXMakeContextCurrentARB(displayPtr_,DefaultRootWindow(displayPtr_),DefaultRootWindow(displayPtr_),context_);
}


/**
 * @copydoc GLContextInterface::releaseCurrent()
 */
bool GLContext::releaseCurrent() const {
    if (isCurrent()) {
        return glXMakeContextCurrentARB(displayPtr_, None, None, nullptr);
    } else return false;
}


/**
 * @copydoc GLContextInterface::sync()
 */
void GLContext::sync() const {
    glFinish();
    glXSwapBuffers(displayPtr_,DefaultRootWindow(displayPtr_));
}


/**
 * @copydoc GLContextInterface::useDefaultSurface()
 */
void GLContext::useDefaultSurface() {
    makeCurrent();
}


/**
 * @copydoc GLContextInterface::isDerivedFrom()
 */
bool GLContext::isDerivedFrom(const GLContextInterface * main) const {
    return (static_cast<const GLContextInterface *>(derivedFrom_) == main);
}


/**
 * @copydoc GLContextInterface::hash()
 */
uint64_t GLContext::hash() const {
  return (uint64_t )context_;
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Prepare X11 for multi-threaded use
 *
 * Instruct X11 that we are about to do some multi-threaded OpenGL operations.
 *
 * @warning This function \b must be called before any other call to X routines and it \b must be
 *          called from the <b>main thread</b>.
 */
void GLContext::initMultiThreading() {
    // Tell X11 we will be doing multi-threaded stuff
    bool expect_false = false;
    if (THREAD_INIT.compare_exchange_strong(expect_false, true)) {
        Status s = XInitThreads();
        assert(s != 0);
    }
}
#endif


/**
 * @copydoc GLContextInterface::getWritePBOPool
 */
PBOPool * GLContext::getWritePBOPool() const {
    if (isDerived()) {
        auto * main = getMain();
        if (!main) THROW_EXCEPTION_ARGS(GLException,"No main context registered");
        return main->getWritePBOPool();
    } else {
        assert(manager_);
        return manager_->getWritePBOPool();
    }
}

/**
 * @copydoc GLContextInterface::getReadPBOPool
 */
PBOPool * GLContext::getReadPBOPool() const {
    if (isDerived()) {
        auto * main = getMain();
        if (!main) THROW_EXCEPTION_ARGS(GLException,"No main context registered");
        return main->getReadPBOPool();
    } else {
        assert(manager_);
        return manager_->getReadPBOPool();
    }
}

/**
 * @copydoc GLContextInterface::getMain()
 */
GLContextInterface * GLContext::getMain() const {
    assert(manager_);
    return manager_->getMain();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Create a wrapped GL context from the currently active context
 *
 * @param idx Index that will be used by the GfxContextManager to store the new context
 * @param mgr Pointer to GfxContextManager that manages this context
 *
 * @return Pointer to OpenGL context object
 *
 * This function wraps the currently active GL context into this wrapper class, such that
 * it can be used with the context manager, for example to create shared contexts.
 */
GLContext * GLContext::createFromCurrent(int idx, fyusenet::GfxContextManager *mgr) {
    GLXContext extctx = glXGetCurrentContext();
    if (!extctx) return nullptr;
    GLContext * ctx = new GLContext(extctx, idx, mgr);
    ctx->displayPtr_ = XOpenDisplay(nullptr);
    ctx->external_ = true;
    return ctx;
}


/**
 * @brief Derive shared GL context from current context
 *
 * @param idx Index for the new context as determined by the GfxContextManager
 * @param dIdx Derived index as determined by the GfxContextManager
 *
 * @return Pointer to derived GL context that is shared with the current context
 *
 * This function creates a new GLContext instance that has shared resources with the context
 * represented by this object.
 */
GLContext * GLContext::derive(int idx, int dIdx) const {
    assert(manager_);
    if (!context_) THROW_EXCEPTION_ARGS(GLException,"Cannot derive context from empty context");
    int numconfigs = 0;
    GLXFBConfig *fbconfigs = glXChooseFBConfig(displayPtr_, DefaultScreen(displayPtr_), visualAttribs, &numconfigs);
    if (numconfigs <= 0) THROW_EXCEPTION_ARGS(GLException,"Desired configuration not available");
    GLXContext newctx = glXCreateContextAttribsARB(displayPtr_, fbconfigs[0], context_, True, context_attribs);
    if (!newctx) THROW_EXCEPTION_ARGS(GLException,"Unable to derive context");
    GLContext * result = new GLContext(newctx, this, idx, dIdx, manager_);
    result->displayPtr_ = displayPtr_;
    return result;
}


/**
 * @brief Constructor for shared context
 *
 * @param ctx GLXContext to wrap
 * @param from Context that this context is derived from (shared with)
 * @param idx Context index as determined by the GfxContextManager
 * @param dIdx Derived context index as determined by the GfxContextManager
 * @param mgr Pointer to GfxContextManager that manages this and the \p from context
 */
GLContext::GLContext(GLXContext ctx, const GLContext * from, int idx, int dIdx, fyusenet::GfxContextManager *mgr) :
    GLContextInterface(idx, from->device()), context_(ctx), derivedFrom_(from), manager_(mgr) {
    derivedIdx_ = dIdx;
}

/**
 * @brief Create context wrapper from external GLX context
 *
 * @param ctx External GLX context to wrap
 * @param idx Internal context index to use for the context
 * @param mgr Pointer to GfxContextManager that manages this context
 */
GLContext::GLContext(GLXContext ctx, int idx, fyusenet::GfxContextManager * mgr) : GLContextInterface(idx, 0), context_(ctx), manager_(mgr) {
    if (!glXCreateContextAttribsARB) {
        glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
        glXMakeContextCurrentARB   = (glXMakeContextCurrentARBProc)glXGetProcAddressARB((const GLubyte *)"glXMakeContextCurrent");
    }
}


} // opengl namespace
} // fyusion namespace

#endif

// vim: set expandtab ts=4 sw=4:
