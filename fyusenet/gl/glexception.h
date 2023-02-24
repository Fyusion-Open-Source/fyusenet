//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// General OpenGL Exception (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/fynexception.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {


CUSTOM_EXCEPTION(GLException,fyusion::FynException);
CUSTOM_EXCEPTION(GLNotImplException, GLException);

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
