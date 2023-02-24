//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Vertex Shader Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <string>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "shader.h"
#include "shaderresource.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

/**
 * @brief Class wrapper for vertex shaders
 *
 * This class specializes the Shader class, please see the documentation there.
 *
 * @see Shader
 * @see https://www.khronos.org/opengl/wiki/Vertex_Shader
 */
class VertexShader : public Shader {
 public:
    /**
     * @brief Constructor
     *
     * @param context GL context that the shader should work with
     *
     * Idle constructor.
     *
     * @note It is recommended to create new shaders by either using #fromString or #fromResource
     */
    VertexShader(const fyusenet::GfxContextLink& context = fyusenet::GfxContextLink()) : Shader(GL_VERTEX_SHADER, context) {
    }

    /**
     * @brief Construct object with source code
     *
     * @param code Pointer to source code for the shader
     *
     * @param context GL context that the shader should work with
     *
     * Constructor that initializes the code part with the supplied source code. No compilation is done.
     *
     * @note It is recommended to create new shaders by either using #fromString or #fromResource
     */
    VertexShader(const char * code, const fyusenet::GfxContextLink& context = fyusenet::GfxContextLink()) : Shader(GL_VERTEX_SHADER, context) {
        setCode(code);
    }

    /**
     * @brief Create vertex shader from source code
     *
     * @param code Pointer to source code for the shader
     * @param context GL context that the shader should work with
     *
     * @return Shared pointer to vertex shader
     *
     * Creates a new vertex shader object and initializes it with the supplied code. No compilation
     * is done.
     */
    static shaderptr fromString(const char *code, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink()) {
        return shaderptr(new VertexShader(code, context));
    }

    /**
     * @brief Create vertex shader from shader resource
     *
     * @param resName Pointer to resource name to read shader from
     * @param context GL context that the shader should work with
     *
     * @return Shared pointer to vertex shader
     *
     * Creates a new vertex shader object by using the ShaderRepository and the supplied \p resName
     * to retrieve shader code from the repository. No compilation of the shader is done.
     */
    static shaderptr fromResource(const char *resName, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink()) {
        const char * code = ShaderRepository::getShader(resName);
        assert(code);
        return shaderptr(new VertexShader(code, context));
    }
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
