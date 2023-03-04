//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Shader Exception (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "glexception.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

CUSTOM_EXCEPTION(ShaderException, GLException);

} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
