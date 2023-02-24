//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Uniform Buffer Object (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "glbuffer.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

/**
 * @brief Wrapper class for OpenGL Uniform-Buffer-Objects (UBOs)
 *
 * This class wraps a uniform-buffer-object. UBOs can be used to store a (larger) set of
 * uniforms in a buffer which can be passed into shader programs quickly via interface
 * blocks.
 *
 * @see https://www.khronos.org/opengl/wiki/Uniform_Buffer_Object
 * @see https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)
 */
class UBO : public GLBuffer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    UBO(const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());
    UBO(GLuint handle, const fyusenet::GfxContextLink &context = fyusenet::GfxContextLink());

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void bindTo(int bindingIndex);
    void bindRangeTo(int bindingIndex,int offset, int size);
};


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
