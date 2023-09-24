//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context for CGL
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../common/logging.h"
#include "../glexception.h"
#include "../glcontext.h"
#include "../../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------

#ifndef __APPLE__
#error This file should not be compiled
#else

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------


static CGLPixelFormatAttribute pixelFormatAttributes[] = {
    kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute) kCGLOGLPVersion_GL4_Core,
    kCGLPFAColorSize, (CGLPixelFormatAttribute) 32,
    kCGLPFAAlphaSize, (CGLPixelFormatAttribute) 8,
    kCGLPFADepthSize, (CGLPixelFormatAttribute) 24,
    kCGLPFAStencilSize, (CGLPixelFormatAttribute) 8,
    kCGLPFAAccelerated,
    kCGLPFADoubleBuffer,
    kCGLPFASampleBuffers, (CGLPixelFormatAttribute) 1,
    kCGLPFASamples, (CGLPixelFormatAttribute) 4,
    (CGLPixelFormatAttribute) 0,
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
GLContext::GLContext(int idx, int device, fyusenet::GfxContextManager *manager, int width, int height) :
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
    CGLSetCurrentContext(nullptr);
    CGLDestroyContext(context_);
    context_ = nullptr;
}



/**
 * @copydoc GLContextInterface::isCurrent()
 */
bool GLContext::isCurrent() const {
    if (!context_) return false;
    CGLContextObj ctx = CGLGetCurrentContext();
    if (!ctx) return false;
    return (ctx == context_);
}


/**
 * @copydoc GLContextInterface::init()
 */
void GLContext::init() {
    CGLPixelFormatObj pixelFormat;
    GLint numberOfPixels;
    CGLChoosePixelFormat(pixelFormatAttributes, &pixelFormat, &numberOfPixels);
    CGLCreateContext(pixelFormat, 0, &context_);
    CGLDestroyPixelFormat(pixelFormat);
}


/**
 * @copydoc GLContextInterface::makeCurrent()
 */
bool GLContext::makeCurrent() const {
    CGLSetCurrentContext(context_);
    return true;
}


/**
 * @copydoc GLContextInterface::releaseCurrent()
 */
bool GLContext::releaseCurrent() const {
    if (isCurrent()) {
        CGLSetCurrentContext(nullptr);
        return true;
    } else return false;
}


/**
 * @copydoc GLContextInterface::sync()
 */
void GLContext::sync() const {
    glFinish();
    // TODO (mw) buffer swap ?
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
bool GLContext::isDerivedFrom(const GLContextInterface * other) const {
    return (static_cast<const GLContextInterface *>(derivedFrom_) == other);
}


/**
 * @copydoc GLContextInterface::hash()
 */
uint64_t GLContext::hash() const {
  return (uint64_t )context_;
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
    assert(manager_);
    if (!context_) THROW_EXCEPTION_ARGS(GLException,"Cannot derive context from empty context");
    CGLPixelFormatObj pixelFormat;
    GLint numberOfPixels;
    CGLChoosePixelFormat(pixelFormatAttributes, &pixelFormat, &numberOfPixels);
    CGLContextObj newctx = 0;
    CGLCreateContext(pixelFormat, context_, &newctx);
    CGLDestroyPixelFormat(pixelFormat);
    if (!newctx) THROW_EXCEPTION_ARGS(GLException,"Unable to derive context");
    return new GLContext(newctx, this, idx, dIdx, manager_);
}


/**
 * @brief Constructor for shared context
 *
 * @param ctx CGLContext to wrap
 * @param from Context that this context is derived from (shared with)
 * @param idx Context index as determined by the GfxContextManager
 * @param dIdx Derived context index as determined by the GfxContextManager
 * @param mgr Pointer to GfxContextManager that manages this and the \p from context
 */
GLContext::GLContext(CGLContextObj ctx, const GLContext * from, int idx, int dIdx, fyusenet::GfxContextManager * mgr) :
    GLContextInterface(idx, from->device()), context_(ctx), derivedFrom_(from), manager_(mgr) {
    derivedIdx_ = dIdx;
}


} // opengl namespace
} // fyusion namespace

#endif

// vim: set expandtab ts=4 sw=4:
