//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Index Buffer Object (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "glbuffer.h"
#include "../gpu/gfxcontextlink.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

/**
 * @brief Simple wrapper for an element array buffer (a.k.a. index buffer) object
 *
 * This class wraps an element array buffer which basically contains indices for rendering
 * geometries using the fixed-function GL geometry part.
 *
 * @see https://www.khronos.org/opengl/wiki/Buffer_Object
 */
class IBO : public GLBuffer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    IBO(const fyusenet::GfxContextLink &context = fyusenet::GfxContextLink());
    IBO(GLuint handle, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
