//--------------------------------------------------------------------------------------------------
// Project: FyuseNet
//--------------------------------------------------------------------------------------------------
// Module : Compute Shader Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "shader.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------



/**
 * @brief Class wrapper for fragment shaders
 *
 * This class specializes the Shader class, please see the documentation there.
 *
 * @see Shader
 * @see https://www.khronos.org/opengl/wiki/Compute_Shader
 */
class ComputeShader : public Shader {
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
    ComputeShader(const GfxContextLink& context = GfxContextLink()) : Shader(GL_COMPUTE_SHADER, context) {
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
    ComputeShader(const char * code,const GLContextLink& context = GfxContextLink()) : Shader(GL_COMPUTE_SHADER, context) {
        setCode(code);
    }

    /**
     * @brief Create compute shader from source code
     *
     * @param code Pointer to source code for the shader
     * @param context GL context that the shader should work with
     *
     * @return Shared pointer to compute shader
     *
     * Creates a new compute shader object and initializes it with the supplied code. No compilation
     * is done.
     */
    static shaderptr fromString(const char *code, const GfxContextLink & context = GfxContextLink()) {
        return shaderptr(new ComputeShader(code, context));
    }

    /**
     * @brief Create compute shader from shader resource
     *
     * @param resName Pointer to resource name to read shader from
     * @param context GL context that the shader should work with
     *
     * @return Shared pointer to compute shader
     *
     * Creates a new compute shader object by using the ShaderRepository and the supplied \p resName
     * to retrieve shader code from the repository. No compilation of the shader is done.
     */
    static shaderptr fromResource(const char *resName, const GfxContextLink & context = GfxContextLink()) {
        const char * code = ShaderRepository::getShader(resName);
        assert(code);
        return shaderptr(new ComputeShader(code, context));
    }

};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
