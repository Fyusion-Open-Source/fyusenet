//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Index Buffer Object
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------


#include "ibo.h"

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
 * @param context GL context that this buffer operates under
 */
IBO::IBO(const fyusenet::GfxContextLink& context) : GLBuffer(GL_ELEMENT_ARRAY_BUFFER,context) {
}


/**
 * @brief Constructor
 *
 * @param handle Existing GL handle to wrap around (this object takes ownership)
 * @param context GL context that this buffer operates under
 */
IBO::IBO(GLuint handle, const fyusenet::GfxContextLink & context) :
    GLBuffer(GL_ELEMENT_ARRAY_BUFFER, handle, false, context) {
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
