//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// EGL Helper Routines (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

#ifndef FYUSENET_USE_EGL
#error THIS FILE SHOULD NOT BE INCLUDED
#endif

//--------------------------------------- System Headers -------------------------------------------

#include <memory>
#include <functional>
#include <EGL/egl.h>
#include <EGL/eglext.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gpu/gfxcontextlink.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------


class EGLHelper {
 public:
    static void iterateEGLDisplays(std::function<void(EGLDisplay eglDisplay, bool* stop)> f);
 private:
    static void initEGLExtensions();
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
