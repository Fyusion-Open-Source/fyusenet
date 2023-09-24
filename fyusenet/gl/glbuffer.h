//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Generic OpenGL Buffer Object (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "../gpu/gfxcontextlink.h"
#include "../gpu/gfxcontexttracker.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

/**
 * @brief Base class for OpenGL buffer objects of various kinds
 *
 * This class serves as base for various types of GL buffer objects like vertex buffers, pixel
 * buffers or index buffers. It tracks the GL context it was created under and when used in debug
 * mode, it performs additional sanity checks that aid in the detection of programming errors.
 */
class GLBuffer : public fyusenet::GfxContextTracker {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    GLBuffer(GLenum target,const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());
    GLBuffer(GLenum target,GLuint handle,bool bound=false, const fyusenet::GfxContextLink &context = fyusenet::GfxContextLink());
    virtual ~GLBuffer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void bind();
    void bind(GLenum target);
    void unbind();
    void unbind(GLenum target);
    void setBufferData(void *data, int dataSize, GLenum usage);
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void genBuffer();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GLenum target_;                     //!< Default GL target to bind the buffer to
    GLuint handle_;                     //!< Buffer handle (GL)
    bool bound_;                        //!< Indicator if buffer is bound or not
};


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
