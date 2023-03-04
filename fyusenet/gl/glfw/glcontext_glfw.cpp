//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context for GLFW
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#ifdef FYUSENET_USE_GLFW

//--------------------------------------- System Headers -------------------------------------------

#ifndef __APPLE__
#include <GLFW/glfw3.h>
#include <GL/glx.h>
#else
#include <GL/glfw.h>
#endif

#include <atomic>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../common/logging.h"
#include "../glexception.h"
#include "../glcontext.h"
#include "../../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------

#ifdef FYUSENET_USE_GLFW

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------

static std::atomic<bool> THREAD_INIT{false};
static std::atomic<bool> GLFW_INIT{false};

static void errorCallback(int error, const char *message);

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
}


/**
 * @brief Destructor
 *
 * Takes down GL context and releases resources held by it.
 */
GLContext::~GLContext() {    
#ifdef DEBUG
    if (uses() > 0) {
        FNLOGW("Destroying GL context with %d active links, check your code", uses());
    }
#endif
    if (context_) glfwDestroyWindow(context_);
    context_ = nullptr;
}


/**
 * @copydoc GLContextInterface::isCurrent()
 */
bool GLContext::isCurrent() const {
    return true;
}


/**
 * @copydoc GLContextInterface::makeCurrent()
 */
bool GLContext::makeCurrent() const {
    glfwMakeContextCurrent(context_);
    return true;
}


/**
 * @copydoc GLContextInterface::releaseCurrent()
 */
bool GLContext::releaseCurrent() const {
    if (isCurrent()) {
        glfwMakeContextCurrent(nullptr);
        return true;
    } else return false;
}


/**
 * @copydoc GLContextInterface::sync()
 */
void GLContext::sync() const {
    glFinish();
    glfwSwapBuffers(context_);
}



/**
 * @copydoc GLContextInterface::init()
 */
void GLContext::init() {
    bool expect = false;
    if (GLFW_INIT.compare_exchange_strong(expect,true)) {
        glfwInit();
        glfwSetErrorCallback(errorCallback);
    }
    context_ = glfwCreateWindow(width_, height_, "mtnwrw", nullptr, nullptr);
    if (!context_) THROW_EXCEPTION_ARGS(GLException, "Cannot initialize GLFW window");
    glfwMakeContextCurrent(context_);
}


/**
 * @copydoc GLContextInterface::useDefaultSurface()
 */
void GLContext::useDefaultSurface() {
    makeCurrent();
}


/**
 * @copydoc GLContextInterface::hash()
 */
uint64_t GLContext::hash() const {
    return (uint64_t)context_;
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
 * @copydoc GLContextInterface::isDerivedFrom()
 */
bool GLContext::isDerivedFrom(const GLContextInterface * other) const {
    return (static_cast<const GLContextInterface *>(derivedFrom_) == other);
}


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
    THROW_EXCEPTION_ARGS(GLNotImplException,"Not implemented yet");
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
    if (!context_) THROW_EXCEPTION_ARGS(GLException,"Cannot derive context from empty context");
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow * win = glfwCreateWindow(32, 32, "hidden", nullptr, context_);
    return new GLContext(win, this, idx, dIdx, manager_);
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
GLContext::GLContext(GLFWwindow *win, const GLContext * from, int idx, int dIdx, fyusenet::GfxContextManager * mgr) :
    GLContextInterface(idx, from->device()), context_(win), derivedFrom_(from), manager_(mgr) {
    derivedIdx_ = dIdx;
}

void errorCallback(int error, const char *message) {
    FNLOGE("GLFW error (%d): %s\n", error, message);
}

} // opengl namespace
} // fyusion namespace

#endif
#endif
// vim: set expandtab ts=4 sw=4:
