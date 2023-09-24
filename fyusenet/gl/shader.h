//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "../gpu/gfxcontextlink.h"
#include "../gpu/gfxcontexttracker.h"
#include "glinfo.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::opengl {

/**
 * @brief Wrapper class for OpenGL shaders
 *
 * This class wraps a GLSL shader where the exact type of shader is determined by subclassing.
 * Shaders themselves are not executable and need to be aggregated by a ShaderProgram instance
 * for actual usage.
 *
 * In order to create a Shader object, use one of the subclassed shader wrappers and just supply the
 * source-code. For example:
 *
 * @code
 * shaderptr shader = VertexShader::fromString(shaderCode);
 * shader.setPreprocDefs("#define MYDEF 1\n");
 * shader.compile()
 * @endcode
 *
 * This snippet creates a vertex shader from a simple string, adds a preprocessor definition, which is
 * automatically inserted after the version directive, and compiles the shader. Note that the compilation
 * can also be done after aggregating the shader in a ShaderProgram instance.
 *
 * This wrapper requires shader code to be in a slightly different format than what would be
 * expected from a standard GLSL shader. The following example shows a GLSL vertex shader as it
 * would be normally used:
 *
 * @code
 * #version 300 es
 *
 * in highp vec4 vertexAtt;
 * out highp vec2 texCoord;
 *
 *
 * void main(void) {
 *     gl_Position = vec4(vertexAtt.x, vertexAtt.y, 0.0, 1.0);
 *     texCoord = vertexAtt.zw;
 * }
 * @endcode
 *
 * The first line is the version directive for the shader, which specifies the version of GLSL
 * to use for this code. This directive <b>must not</b> be present in any shader code that is
 * supplied to this wrapper and its subclasses, because it will be automatically generated,
 * depending on the platform that was found.
 *
 * In case you use instructions that are specific to a GLSL version, please either use the GLInfo
 * object to check if the GLSL version is correct or recent enough, or provide conditional
 * compilation based on the built-in \c __VERSION__ preprocessor definition in the shader.
 *
 * The following extra definitions are automatically issued in all shaders:
 *  * \c GLES in case the shader is running under OpenGL/ES
 *  * \c BINDING_SUPPORT if a GL/GLSL version is encountered that supports interface binding
 *
 * For additional convenience, this class offers two functionalities that are usually not
 * found in baseline GLSL:
 *  1. Add extra definitions after the (implicit) version directive
 *  2. Ability to include shader portions, called \e snippets, using an \c \#include directory in
 *     the GLSL code itself.
 *
 * The first functionality is already demonstrated in the first code snippet above and results
 * in the extra defs being pasted directly into the shader code (after the version directive)
 * before compiling it.
 *
 * For the 2nd functionality, we supply a simple example code snippet:
 *
 * @code
 * precision highp float;
 * precision highp int;
 * precision highp sampler2D;
 *
 * layout(location=0) out vec4 fragment;
 * in vec2 texCoord;
 *
 * uniform sampler2D srcData;
 *
 * #include "shaders/activation.inc"
 *
 * void main(void) {
 *     fragment = relu(texture(srcData,texCoord));
 * }
 * @endcode
 *
 * The above example includes a shadersnippet - in this case a snippet that contains the definition
 * of the \c relu() function - by making use of the ShaderRepository and ShaderResource classes.
 *
 * @see https://www.khronos.org/opengl/wiki/Shader
 */
class Shader : public fyusenet::GfxContextTracker {
    friend class ShaderProgram;
    friend class ShaderCache;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    Shader(GLenum type, const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink(), GLInfo::glslver = GLInfo::UNSPECIFIED);
    virtual ~Shader();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    GLenum getType() const;
    void release();
    void setPreprocDefs(const char *defs);
    void setPreprocDefs(const std::string& defs);
    void setCode(const char *data);
    void setCode(const std::string& data);
    void setResourceName(const std::string& name);
    std::string getCode() const;
#ifndef ANDROID
    void loadFromFile(const char *fileName);
#endif
    void compile();
    bool isCompiled() const;
    void log() const;

    /**
     * @brief Get underlying OpenGL shader handle
     *
     * @return Shader handle or 0 if shader is not valid
     */
    GLuint getHandle() const {
      return handle_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void compile(const char *data);
    void logError() const;
    void logShader(const char *data) const;
    std::string includeSnippets(const std::string& code);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::string preamble_;                           //!< Generated preamble (version string)
    std::string shaderCode_;                         //!< Actual shader source code (with include statements resolved)
    std::string preprocDefs_;                        //!< Additional preprocessor definitions following the preamble
    std::string resourceName_;                       //!< Optional resource name that this shader was created from
    GLuint handle_ = 0;                              //!< OpenGL handle for the shader (valid after successful compilation)
    GLenum type_ = 0;                                //!< Shader type (e.g. fragment shader, vertex shader, etc.)
    GLInfo::glslver version_ = GLInfo::UNSPECIFIED;  //!< Target GLSL version for the shader, if left at UNSPECIFIED, most recent platform version will be used
    mutable uint64_t hash_;                          //!< Hash that is computed over the (full) shader code for caching and computed externally
};


typedef std::shared_ptr<Shader> shaderptr;

} // fyusion::opengl namespace



// vim: set expandtab ts=4 sw=4:
