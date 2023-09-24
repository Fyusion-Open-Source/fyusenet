//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// OpenGL Context for WGL                                                      (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winuser.h>
#include <wingdi.h>
#include <GL/glew.h>
#include <GL/wglew.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../glexception.h"
#include "../../common/logging.h"
#include "../glcontext.h"
#include "../../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::opengl {

//-------------------------------------- Local Definitions -----------------------------------------

//#define SHOW_WINDOW

static const char * WINCLASSNAME = "fyusenet_class";

static const PIXELFORMATDESCRIPTOR PIXELFORMAT = {
        sizeof(PIXELFORMATDESCRIPTOR ), 1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL,
        PFD_TYPE_RGBA,
        32, 0, 0, 0, 0, 0, 0,
        8, 0, 0, 0, 0, 0,0,
        24,
        8, 0, 0, 0, 0, 0, 0
};


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
GLContext::GLContext(int idx, int device, fyusenet::GfxContextManager * manager, int width,int height) :
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
    wglMakeCurrent(nullptr, nullptr);
    if (!external_) {
        if (context_) wglDeleteContext(context_);
        context_ = nullptr;
        if (device_) ReleaseDC(window_, device_);
        if (window_) DestroyWindow(window_);
        if (instance_) UnregisterClass(TEXT(WINCLASSNAME), instance_);
    }
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
 * @copydoc GLContextInterface::isCurrent()
 */
bool GLContext::isCurrent() const {
    if (!context_) return false;
    return (context_ == wglGetCurrentContext());
}


/**
 * @copydoc GLContextInterface::init()
 */
void GLContext::init() {
    WNDCLASS wc = {0};
    instance_ = GetModuleHandle(nullptr);
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = instance_;
    wc.lpszClassName = TEXT(WINCLASSNAME);
    if (!RegisterClass(&wc)) {
        THROW_EXCEPTION_ARGS(GLException, "Cannot register class for dummy window");
    }
    window_ = CreateWindow(wc.lpszClassName, TEXT("FyuseNet Dummy"), 0, 0, 0, 128, 128, nullptr, nullptr, wc.hInstance, nullptr);
    if (window_) {
        if (device_ = GetDC(window_); device_) {
            if (int pixformat = ChoosePixelFormat(device_, &PIXELFORMAT); pixformat) {
                if (SetPixelFormat(device_, pixformat, &PIXELFORMAT)) {
                    context_ = wglCreateContext(device_);
                    if (context_) {
                        makeCurrent();
                        if (GLenum rc = glewInit(); rc != GLEW_OK) {
                            THROW_EXCEPTION_ARGS(GLException,"Cannot initialize GLEW");
                        }
                        if (GLenum rc = wglewInit(); rc != GLEW_OK) {
                            THROW_EXCEPTION_ARGS(GLException,"Cannot initialize wGLEW");
                        }
#ifdef SHOW_WINDOW
                        ShowWindow(window_, SW_SHOW);
#endif
                        wglMakeCurrent(nullptr, nullptr);
                        wglDeleteContext(context_);
                        // create core context now
                        int coreattribs[] = {
                                WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
                                WGL_CONTEXT_MINOR_VERSION_ARB, 3,
                                //WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
                                WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
                        auto core = wglCreateContextAttribsARB(device_, nullptr, coreattribs);
                        if (core) {
                            context_ = core;
                            makeCurrent();
#if 0
                            glEnable(GL_DEBUG_OUTPUT);
                            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
                            glDebugMessageCallback([](GLenum src, GLenum typ, GLuint id, GLenum sev, GLsizei length, const GLchar * msg, const void *user) {
                                printf("OpenGL: %s\n", msg);
                                fflush(stdout);
                            }, nullptr);
#endif
                        } else {
                            THROW_EXCEPTION_ARGS(GLException,"Cannot create core GL context (4.3)");
                        }
                    }
                }
            } else {
                ReleaseDC(window_, device_);
                DestroyWindow(window_);
            }
        } else DestroyWindow(window_);
    }
    if (!context_) {
        window_ = nullptr;
        device_ = nullptr;
        UnregisterClass(TEXT(WINCLASSNAME), instance_);
        THROW_EXCEPTION_ARGS(GLException,"Unable to create context");
    }
}


/**
 * @copydoc GLContextInterface::makeCurrent()
 */
bool GLContext::makeCurrent() const {
    if (external_) {
        return false;
    } else {
        return wglMakeCurrent(device_, context_);
    }
}


/**
 * @copydoc GLContextInterface::releaseCurrent()
 */
bool GLContext::releaseCurrent() const {
    if (isCurrent()) {
        wglMakeCurrent(nullptr, nullptr);
        return true;
    } else return false;
}


/**
 * @copydoc GLContextInterface::sync()
 */
void GLContext::sync() const {
    glFinish();
#ifdef SHOW_WINDOW
    SwapBuffers(device_);
#endif
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


/**
 * @copydoc GLContextInterface::getMain()
 */
GLContextInterface * GLContext::getMain() const {
    assert(manager_);
    return manager_->getMain();
}

/**
 * @copydoc GLContextInterface::texturePool()
 */
ScopedTexturePool * GLContext::texturePool() const {
    assert(manager_);
    return manager_->texturePool();
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
 * @return Pointer to OpenGL context object or \c nullptr if there was no context current
 *
 * This function wraps the currently active GL context into this wrapper class, such that
 * it can be used with the context manager, for example to create shared contexts.
 */
GLContext * GLContext::createFromCurrent(int idx, fyusenet::GfxContextManager *mgr) {
    HDC dev = wglGetCurrentDC();
    HGLRC cctx = wglGetCurrentContext();
    auto * ctx = new GLContext(cctx, dev, idx, mgr);
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
    assert(derivedFrom_ == nullptr);
    if (!context_) THROW_EXCEPTION_ARGS(GLException,"Cannot derive context from empty context");
    int coreattribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
            WGL_CONTEXT_MINOR_VERSION_ARB, 3,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
    HGLRC shared = wglCreateContextAttribsARB(device_, nullptr, coreattribs);
    if (shared) {
        if (wglShareLists(shared, context_)) {
            return new GLContext(shared, this, idx, dIdx, manager_);
        }
    }
    THROW_EXCEPTION_ARGS(GLException,"Cannot create shared context");
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
GLContext::GLContext(HGLRC ctx, const GLContext * from, int idx, int dIdx, fyusenet::GfxContextManager * mgr) :
    GLContextInterface(idx, from->device()), context_(ctx), derivedFrom_(from), manager_(mgr) {
    derivedIdx_ = dIdx;
}


/**
 * @brief Constructor from existing context
 *
 * @param ctx GLXContext to wrap
 * @param device Device context to wrap
 * @param idx Context index as determined by the GfxContextManager
 * @param mgr Pointer to GfxContextManager that manages this and the \p from context
 */
GLContext::GLContext(HGLRC ctx, HDC device, int idx, fyusenet::GfxContextManager * mgr) :
    // FIXME (mw) we always assume a zero device (graphics card) here
    GLContextInterface(idx, 0), device_(device), context_(ctx), manager_(mgr) {
}


} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
