//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context Abstraction (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

#ifndef FYUSENET_INTERNAL
#cmakedefine FYUSENET_USE_EGL
#cmakedefine FYUSENET_USE_GLFW
#cmakedefine FYUSENET_USE_WEBGL
#ifndef FYUSENET_MULTITHREADING
#cmakedefine FYUSENET_MULTITHREADING
#endif
#endif

#if defined(FYUSENET_USE_EGL)
// headers for Android and/or GLES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#ifndef ANDROID
#include <GLES3/gl32.h>
#endif
#endif

#if defined(__linux__) && !defined(FYUSENET_USE_EGL)
// headers for linux
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#ifdef FYUSENET_USE_GLFW
#include <GLFW/glfw3.h>
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glext.h>
#endif

#ifdef __APPLE__
// headers for MacOS
#include <OpenGL/CGLTypes.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLContext.h>
#include <OpenGL/gl3.h>
#include <OpenGL/OpenGL.h>
#endif
#ifdef FYUSENET_USE_WEBGL
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgl.h>
#include <GLES3/gl3.h>
#endif


//-------------------------------------- Project  Headers ------------------------------------------

#include "glcontextinterface.h"

class TestContextManager;

namespace fyusion {

namespace fyusenet {
    class GfxContextManager;
}

namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------


/**
 * @brief Realization / encapsulation of an OpenGL(ES) context
 *
 * This class is the actual realization of the OpenGL(ES) context that is system-specific.
 * The declaration is more-or-less uniform for all platforms, only the private data and the
 * implementations are system specific.
 *
 * Though this class may be used to pass around a GL contexts to other classes, we advise against
 * doing so and use the GfxContextLink class for this purpose instead. It is even more lightweight,
 * (more) system independent and includes some reference counting which might come in handy.
 *
 * @see GfxContextLink, GLContextInterface
 */
class GLContext : public GLContextInterface {
    friend class fyusenet::GfxContextManager;
    friend class ::TestContextManager;
#ifdef FYUSENET_USE_EGL
    friend class EGLHelper;
#endif
 public:
    enum {
#if defined(FYUSENET_USE_GLFW)
        DEFAULT_WIDTH = 256,
        DEFAULT_HEIGHT = 256
#else
        DEFAULT_WIDTH = 32,
        DEFAULT_HEIGHT = 32
#endif
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    GLContext(int idx, int device, fyusenet::GfxContextManager * manager, int width=DEFAULT_WIDTH, int height=DEFAULT_HEIGHT);
#ifdef FYUSENET_USE_WEBGL
    GLContext(char *canvasID, int idx, fyusenet::GfxContextManager * manager, int width, int height);
#endif
    virtual ~GLContext();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual PBOPool * getWritePBOPool() const override;
    virtual PBOPool * getReadPBOPool() const override;
    virtual bool makeCurrent() const override;
    virtual bool releaseCurrent() const override;
    virtual void init() override;
    virtual void sync() const override;
    virtual void useDefaultSurface() override;
    virtual bool isCurrent() const override;
    virtual bool isDerivedFrom(const GLContextInterface * main) const override;
    virtual uint64_t hash() const override;
    virtual GLContextInterface * getMain() const override;

#if !defined(FYUSENET_USE_EGL) && defined(__linux__) && defined(FYUSENET_MULTITHREADING)
    static void initMultiThreading();
#endif

    /**
     * @brief Check if context object wraps an external context or an internally managed one
     *
     * @retval true if context object wraps an externally supplied GL context
     * @retval false if context object wraps a managed GL context
     */
    bool isExternal() const {
        return external_;
    }

#ifdef FYUSENET_USE_GLFW
    /**
     * @brief Check if context matches a GLFW window
     *
     * @param ctx GLFWwindow instance to check for equality to this context
     *
     * @retval true if context matches the window
     * @retval false otherwise
     */
    bool matches(GLFWwindow *ctx) const {
        return (ctx == context_);
    }

    /**
     * @brief Get pointer to GLFW window
     *
     * @return GLFW window
     */
    GLFWwindow * window() const {
        return context_;
    }
#else // not GLFW
#if defined(__linux__) && !defined(FYUSENET_USE_EGL)
    /**
     * @brief Check if context matches a GLX context
     *
     * @param ctx GLXContext to check for equality to this context
     *
     * @retval true if context matches the supplied GLX context
     * @retval false otherwise
     */
    bool matches(GLXContext ctx) const {
        return (ctx == context_);
    }
#else // not GLX
#ifdef FYUSENET_USE_EGL
    /**
     * @brief Check if context matches EGL context
     *
     * @param ctx EGLContext to check for equality to this context
     *
     * @retval true if context matches the supplied GLX context
     * @retval false otherwise
     */
    bool matches(EGLContext ctx) const {
        return (ctx == context_);
    }
#elif defined(FYUSENET_USE_WEBGL) // WebGL
    /**
     * @brief Check if context matches WebGL context
     *
     * @param ctx WebGL context to check for equality to this context
     *
     * @retval true if context matches the supplied GLX context
     * @retval false otherwise
     */
    bool matches(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx) const {
        return (ctx == context_);
    }
#else // not WebGL
    /**
     * @brief Check if context matches CGL context
     *
     * @param ctx CGLContext to check for equality to this context
     *
     * @retval true if context matches the supplied GLX context
     * @retval false otherwise
     */
    bool matches(CGLContextObj ctx) const {
        return (ctx == context_);
    }
#endif
#endif
#endif
#ifdef ANDROID
    void setNativeWindow(NativeWindowType win);
#endif
 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    static GLContext * createFromCurrent(int idx, fyusenet::GfxContextManager *mgr);
    GLContext * derive(int idx, int dIdx) const;
#ifdef FYUSENET_USE_GLFW
    GLContext(GLFWwindow *win, const GLContext *from, int idx, int dIdx, fyusenet::GfxContextManager *mgr);
#else
#if defined(__linux__) && !defined(FYUSENET_USE_EGL)
    GLContext(GLXContext ctx, int idx, fyusenet::GfxContextManager *mgr);
    GLContext(GLXContext ctx, const GLContext *from, int idx, int dIdx, fyusenet::GfxContextManager *mgr);
#else
#if defined(FYUSENET_USE_EGL)
    GLContext(EGLContext ctx, const GLContext *from, int idx, int dIdx, fyusenet::GfxContextManager *mgr);
    GLContext(EGLContext ctx, int idx, fyusenet::GfxContextManager *mgr);
#elif defined(__APPLE__)
    GLContext(CGLContextObj ctx, const GLContext *from, int idx, int dIdx, fyusenet::GfxContextManager *mgr);
#elif defined(FYUSENET_USE_WEBGL)
    GLContext(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx, const GLContext *from, int idx, int dIdx, fyusenet::GfxContextManager *mgr);
#endif
#endif
#endif
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
#if defined(FYUSENET_USE_EGL)
    EGLDisplay display_ = EGL_NO_DISPLAY;               //!< Underlying EGL display
    EGLContext context_ = EGL_NO_CONTEXT;               //!< Underlying EGL context
    EGLConfig activeConfig_ = EGL_NO_CONFIG_KHR;        //!< Display configuration for the context
    EGLSurface defaultSurface_ = EGL_NO_SURFACE;        //!< Default surface for EGL display
    EGLSurface activeSurface_ = EGL_NO_SURFACE;         //!< Currently used surface
#ifdef ANDROID
    NativeWindowType nativeWindow_ = 0;                 //!< Android only
#else
    EGLStreamKHR stream_ = EGL_NO_STREAM_KHR;           //!< For future extensions
#endif
#else // not EGL
#if defined(__linux__) && !defined(FYUSENET_USE_EGL) && !defined(FYUSENET_USE_GLFW)
    Display *displayPtr_ = nullptr;                     //!< Pointer to X display
    GLXContext context_ = 0;                            //!< Underlying GLX context
    GLXPbuffer pBuffer_ = 0;                            //!< Underlying pixelbuffer (surface)
#elif defined(__APPLE__)
    CGLContextObj context_;                             //!< Underlying CGL context
#elif defined(FYUSENET_USE_WEBGL)
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_ = 0;       //!< Underlying WebGL context
    char *canvasID_ = nullptr;                          //!< Target HTML canvas name
    int version_ = 1;                                   //!< WebGL (major) version to use
#endif
#endif
#ifdef FYUSENET_USE_GLFW
    GLFWwindow * context_ = nullptr;                    //!< Underlying GLFW window (context)
#endif
    const GLContext * derivedFrom_ = nullptr;           //!< For derived contexts, this points to the main context
    int width_ = 0;                                     //!< Width of the surface
    int height_ = 0;                                    //!< Height of the surface
    mutable std::atomic<int> derivedCounter_{0};        //!< Counter for the number of derived contexts from this context
    bool external_ = false;                             //!< Indicator if this object wraps an externally supplied context
    fyusenet::GfxContextManager * manager_ = nullptr;   //!< Pointer to context manager object that manages this context
};

} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
