//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Shader-(Pair) Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "shaderprogram.h"
#include "../gpu/gfxcontextlink.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Convenience class to compile and link vertex/fragment shader pairs
 *
 * This class provides a single small interface to compile and link vertex/fragment shader pairs
 * from the resource system.
 *
 * @todo Expand to also include compute shaders in the future
 */
class ShaderBuilder {
 public:
    static programptr shaderProgramFromResource(const char *vertResName, const char *fragResName, const std::type_info& typeInfo, const char *extraDefs = nullptr, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink());
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
