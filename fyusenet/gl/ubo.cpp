//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Uniform Buffer Object
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "ubo.h"
#include "glexception.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


namespace fyusion {
namespace opengl {

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param context Link to GL context
 *
 * Creates an empty UBO object.
 */
UBO::UBO(const fyusenet::GfxContextLink & context) : GLBuffer(GL_UNIFORM_BUFFER, context) {
}


/**
 * @brief Constructor around existing handle
 *
 * @param handle Existing GL handle to be wrapped, no ownership taken
 *
 * @param context Link to GL context
 *
 * Constructs a UBO object around the provided \p handle, ownership is not transferred to this
 * object and will not be deleted from the GL resources on destruction of this object.
 */
UBO::UBO(GLuint handle, const fyusenet::GfxContextLink & context) : GLBuffer(GL_UNIFORM_BUFFER, handle, false, context) {
}


/**
 * @brief Bind %UBO to shader interface
 *
 * @param bindingIndex Interface index to bind to
 */
void UBO::bindTo(int bindingIndex) {
#ifdef DEBUG
    glGetError();
#endif
    bind();
    glBindBufferBase(target_, bindingIndex, handle_);
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Error binding buffer (glerr=0x%x)",err);
#endif
}

/**
 * @brief Bind range of %UBO to shader interface
 *
 * @param bindingIndex Interface index to bind to
 * @param offset Offset (in bytes) to %UBO buffer
 * @param size Size of %UBO portion (in bytes) to map
 *
 * @see https://khronos.org/registry/OpenGL-Refpages/gl4/html/glBindBufferRange.xhtml
 */
void UBO::bindRangeTo(int bindingIndex, int offset, int size) {
#ifdef DEBUG
    glGetError();
#endif
    bind();
    glBindBufferRange(target_, bindingIndex, handle_, offset, size);
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Error binding buffer (glerr=0x%x)",err);
#endif
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
