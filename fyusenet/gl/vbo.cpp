//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Vertex Buffer Object
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "vbo.h"
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
 * Creates an empty VBO object.
 */
VBO::VBO(const fyusenet::GfxContextLink & context) : GLBuffer(GL_ARRAY_BUFFER, context) {
}


/**
 * @brief Constructor around existing handle
 *
 * @param handle Existing GL handle to be wrapped, no ownership taken
 *
 * @param context Link to GL context
 *
 * Constructs a VBO object around the provided \p handle, ownership is not transferred to this
 * object and will not be deleted from the GL resources on destruction of this object.
 */
VBO::VBO(GLuint handle, const fyusenet::GfxContextLink & context):GLBuffer(GL_ARRAY_BUFFER,handle,false,context) {
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
