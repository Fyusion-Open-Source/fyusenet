//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Generic Proxy Geometry Builder (Header)                                     (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <tuple>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../gfxcontextlink.h"

namespace fyusion::fyusenet::gpu::rudiments {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Helper class for generating most common proxy geometries
 *
 * This class provides some boilerplate code to generate the most common proxy geometries. It
 * was introduced after the original code base was done
 * The
 */
class ProxyGenerator {
 public:
    static std::tuple<opengl::VAO *, opengl::VBO *, opengl::IBO *> texturedQuad(const GfxContextLink& context);
    static std::tuple<opengl::VAO *, opengl::VBO *, opengl::IBO *> simpleQuad(const GfxContextLink& context);
    static std::tuple<opengl::VAO *, opengl::VBO *, opengl::IBO *> verticalTexturedQuads(const GfxContextLink& context, const std::vector<int>& heights, int fullHeight);
    static std::tuple<opengl::VAO *, opengl::VBO *> texturedDotMatrix(const GfxContextLink& context, int columns, int rows);
    static std::tuple<opengl::VAO *, opengl::VBO *> dotMatrix(const GfxContextLink& context, int columns, int rows);


 private:
    static const GLfloat texQuadVerts_[];
    static const GLfloat quadVerts_[];
    static const GLshort quadIndices_[];
};

} // fyusion::fyusenet::gpu::rudiments namespace

// vim: set expandtab ts=4 sw=4:

