//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Resource (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <unordered_map>
#include <string>

//-------------------------------------- Project  Headers ------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Repository for shader resource system
 *
 * The shader resource system is a simple compile-time resource manager that collects all shader
 * sources of the project and renders them accessible by a virtual file name.
 *
 * Sbader sources themselves are wrapped by ShaderResource objects internally. The interface
 * to the shaders in the repository always exports them as null-terminated strings however.
 *
 * @see ShaderResource
 */
class ShaderRepository {
    friend class ShaderResource;
 public:
    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    static const char * getShader(const char *resourceName);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    ShaderRepository();
    static ShaderRepository & repository();
    void addResource(const char *shader,const char *resourceName);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::unordered_map<std::string, const char *> shaderMap_;
};


/**
 * @brief Wrapper that pushes a shader resource to the resource system
 *
 * This class merely adds a piece of shader code associated with a resource name to the
 * ShaderRepository
 */
class ShaderResource {
    friend class ShaderRepository;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ShaderResource(const char *shader,const char *resourceName);
};


} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
