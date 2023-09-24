//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context for WebGL
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <emscripten.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl_sys.h"
#include "../glexception.h"
#include "../../common/logging.h"
#include "../glcontext.h"
#include "../../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------

#ifndef FYUSENET_USE_WEBGL
#error This file should not be compiled
#endif

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
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


#ifdef FYUSENET_USE_WEBGL
/**
 * @brief Construct context instance on existing HTML5 canvas
 *
 * @param canvasID Canvas ID in the DOM tree
 * @param idx Internal context index
 * @param manager Pointer to GfxContextManager instance
 * @param width Width of the canvas
 * @param height Height of the canvas
 */
GLContext::GLContext(char *canvasID, int idx, fyusenet::GfxContextManager * manager, int width, int height) :
    GLContextInterface(idx, 0), canvasID_(canvasID), width_(width), height_(height), manager_(manager) {
}
#endif


/**
 * @brief Destructor
 *
 * Takes down GL context and releases resources held by it.
 */
GLContext::~GLContext() {
    emscripten_webgl_destroy_context(context_);
    context_ = 0;
    free(canvasID_);
}


/**
 * @copydoc GLContextInterface::init()
 */
void GLContext::init() {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.explicitSwapControl = EM_FALSE;
    attrs.depth = EM_TRUE;
    attrs.stencil = EM_TRUE;
    attrs.antialias = EM_FALSE;
    attrs.enableExtensionsByDefault = EM_TRUE;
    attrs.majorVersion = 2;
    attrs.minorVersion = 0;
    attrs.enableExtensionsByDefault = EM_TRUE;
    context_ = emscripten_webgl_create_context(canvasID_, &attrs);
    if (context_ <= 0) {
        if (context_ == EMSCRIPTEN_RESULT_UNKNOWN_TARGET) {
            EM_ASM({console.log("Cannot create context, unknown target supplied");});
        } else {
            EM_ASM({console.log("Cannot create context, error "+$0);},context_);
        }
    }
    emscripten_webgl_make_context_current(context_);
    glViewport(0, 0, width_, height_);
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
 * @copydoc GLContextInterface::isDerivedFrom()
 */
bool GLContext::isDerivedFrom(const GLContextInterface * main) const {
    return (static_cast<const GLContextInterface *>(derivedFrom_) == main);
}


/**
 * @copydoc GLContextInterface::makeCurrent()
 */
bool GLContext::makeCurrent() const {
    emscripten_webgl_make_context_current(context_);
    return true;
}


/**
 * @copydoc GLContextInterface::releaseCurrent()
 */
bool GLContext::releaseCurrent() const {
    if (isCurrent()) {
        return emscripten_webgl_make_context_current(0);
    } else return false;
}


/**
 * @copydoc GLContextInterface::isCurrent()
 */
bool GLContext::isCurrent() const {
    if (!context_) return false;
    auto ctx = emscripten_webgl_get_current_context();
    if (!ctx) return false;
    return (ctx == context_);
}


/**
 * @copydoc GLContextInterface::sync()
 */
void GLContext::sync() const {
    emscripten_webgl_commit_frame();
}


/**
 * @copydoc GLContextInterface::hash()
 */
uint64_t GLContext::hash() const {
  return (uint64_t )context_;
}


/**
 * @copydoc GLContextInterface::useDefaultSurface()
 */
void GLContext::useDefaultSurface() {
    makeCurrent();
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
 * @brief Constructor for shared context
 *
 * @param ctx WebGL context to wrap
 * @param from Context that this context is derived from (shared with)
 * @param idx Context index as determined by the GfxContextManager
 * @param dIdx Derived context index as determined by the GfxContextManager
 * @param mgr Pointer to GfxContextManager that manages this and the \p from context
 */
GLContext::GLContext(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx, const GLContext * from, int idx, int dIdx, fyusenet::GfxContextManager * mgr) :
    GLContextInterface(idx, from->device()), context_(ctx), derivedFrom_(from), manager_(mgr) {
    derivedIdx_ = dIdx;
}



} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
