//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Vertex Buffer Object (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "glbuffer.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

/**
 * @brief Vertex buffer object wrapper
 *
 * This class wraps a vertex buffer (a \c GL_ARRAY_BUFFER ) which basically contains vertex positions
 * (and more) and is used to define geometries using the fixed-function GL geometry part.
 *
 * @see https://www.khronos.org/opengl/wiki/Vertex_Specification#Vertex_Buffer_Object
 */
class VBO : public GLBuffer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    VBO(const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());
    VBO(GLuint handle, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());
};


} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
