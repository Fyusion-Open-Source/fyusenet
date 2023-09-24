//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Program (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <unordered_map>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "shader.h"
#include "../gpu/gfxcontexttracker.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::opengl {

class UniformState;
class ShaderProgram;

typedef std::shared_ptr<ShaderProgram> programptr;

/**
 * @brief Aggregate class for aggregating shaders into a shader program
 *
 * This class serves as an aggregate for individual shaders into a shader program. Shaders themselves
 * are not linkable instances, only in conjunction with a ShaderProgram they can be linked into
 * an "executable".
 *
 * To aggregate shaders into a shader program, just add the individual shaders using the #addShader
 * method. Shaders can either be added in compiled or uncompiled form. This class offers a
 * convenience compile() method that will compile all uncompiled shaders. Finally, a shader program
 * can only be used if it has been successfully linked.
 *
 * The following code snippet illustrates the usage:
 * @code
 * auto prog = ShaderProgram::createInstance();
 * prog->addShader(vertexShader);
 * prog->addShader(fragmentShader);
 * prog->compile();
 * prog->link();
 * @endcode
 *
 * %Shader programs cannot be used immediately after linking, they have to be bound first by
 * using the bind() method. Make sure to unbind() a shader after it has been used, as there
 * is an internal flag that keeps track of the bound status.
 *
 * In order to set uniform variables in the shader program, this class offers convenience
 * functionality to set these. This can either be done by simply providing the variable name,
 * or by first mapping the variables to symbols which can be used instead of the names.
 * Especially if the same shader program is used for different parts in the code, it makes sense
 * to keep track of the contents of the uniform variables using a UniformState object.
 *
 * @see UniformState
 * @see https://www.khronos.org/opengl/wiki/Shader
 * @see https://www.khronos.org/opengl/wiki/Uniform_(GLSL)
 */
class ShaderProgram : public fyusenet::GfxContextTracker {
  friend class ShaderCache;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    virtual ~ShaderProgram();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void addShader(shaderptr shader);
    void compile();
    void link();
    void bind(UniformState *state = nullptr);
    void unbind(bool compress=false);
    void log() const;
    GLint mapUniformLocation(const char *name, int symbol, bool optional=false);
    void setMappedUniformValue(int symbol, GLint value, bool optional=false);
    void setMappedUniformValue(int symbol, GLfloat value, bool optional=false);
    void setMappedUniformVec2(int symbol, GLint v0,GLint v1, bool optional=false);
    void setMappedUniformVec2(int symbol, GLfloat v0,GLfloat v1, bool optional=false);
    void setMappedUniformVec3(int symbol, GLint v0, GLint v1, GLint v2, bool optional=false);
    void setMappedUniformVec3(int symbol, GLfloat v0, GLfloat v1, GLfloat v2, bool optional=false);
    void setMappedUniformVec4(int symbol, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, bool optional=false);
    void setMappedUniformVec4Array(int symbol, const GLfloat *data, int num4Entries, bool optional=false);
    void setMappedUniformVec4Array(int symbol, const GLuint *data, int num4Entries, bool optional=false);
    void setMappedUniformMat4(int symbol, const GLfloat *data, bool transpose=false, bool optional=false);
    void setMappedUniformMat4Array(int symbol, const GLfloat *matrices, int numMatrices, bool transpose=false, bool optional=false);
    void setUniformValue(const char *name, GLint value, bool optional=false);
    void setUniformValue(const char *name, GLfloat value, bool optional=false);
    void setUniformValue(GLint location, GLint value);
    void setUniformValue(GLint location, GLfloat value);
    void setUniformVec2(const char *name, GLint v0, GLint v1, bool optional=false);
    void setUniformVec2(const char *name, GLfloat v0, GLfloat v1, bool optional=false);
    void setUniformVec2(GLint location, GLint v0, GLint v1);
    void setUniformVec2(GLint location, GLfloat v0, GLfloat v1);
    void setUniformVec3(GLint location, GLint v0, GLint v1, GLint v2);
    void setUniformVec3(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
    void setUniformVec3(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, bool optional=false);
    void setUniformVec3(const char *name, GLint v0, GLint v1, GLint v2, bool optional=false);
    void setUniformVec4(const char *name, GLint v0, GLint v1, GLint v2, GLint v3, bool optional=false);
    void setUniformVec4(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, bool optional=false);
    void setUniformVec4(GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
    void setUniformVec4(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    void setUniformVec4Array(const char *name, const GLfloat *data, int num4Entries, bool optional=false);
    void setUniformVec4Array(GLint location, const GLfloat *data, int num4Entries);
    void setUniformVec4Array(const char *name, const GLuint *data, int num4Entries, bool optional=false);
    void setUniformVec4Array(GLint location, const GLuint *data,int num4Entries);
    void setUniformVec3Array(const char *name, const GLfloat *data, int num3Entries, bool optional=false);
    void setUniformVec3Array(GLint location, const GLfloat *data, int num3Entries);
    void setUniformVec2Array(const char *name, const GLint *data, int num2Entries, bool optional=false);
    void setUniformVec2Array(GLint location, const GLint *data, int num2Entries);
    void setUniformVec2Array(const char *name, const GLfloat *data, int num2Entries, bool optional=false);
    void setUniformVec2Array(GLint location, const GLfloat *data, int num2Entries);
    void setUniformArray(const char *name, const GLfloat *data,int numEntries, bool optional=false);
    void setUniformArray(GLint location, const GLfloat *data,int numEntries);
    void setUniformMat3(GLint location, const GLfloat *data,bool transpose=false);
    void setUniformMat3(const char *name, const GLfloat *matrix, bool transpose=false, bool optional=false);
    void setUniformMat4(GLint location, const GLfloat *data,bool transpose=false);
    void setUniformMat4(const char *name, const GLfloat *matrix, bool transpose=false, bool optional=false);
    void setUniformMat4Array(const char * name, const GLfloat *matrices, int numMatrices, bool transpose=false, bool optional=false);
    void setUniformMat4Array(GLint location, const GLfloat *matrices, int numMatrices, bool transpose=false);
    void bindAttributeLocation(const char *name, GLuint index);
    void bindIndexToShaderBuffer(const char *bufferName, int bindingIndex);
    GLint resolveLocation(const char *location, bool silent=false) const;
    static programptr createInstance(const fyusenet::GfxContextLink& link = fyusenet::GfxContextLink());

    /**
     * @brief Check if shader program is bound as indicatedd by its internal state flag
     *
     * @retval true Program is currently bound
     * @retval false otherwise
     *
     * @warning In case the internal state is out-of-sync with the actual GL binding, this function
     *          may return a wrong result.
     */
    bool isBound() const {
        return bound_;
    }

    /**
     * @brief Retrieve custom user flags
     *
     * @return 32-bit integer user flags (default is 0)
     */
    unsigned int getUserFlags() const {
        return userFlags_;
    }

    /**
     * @brief Set custom user-defined flags
     *
     * @param flags Flags to
     */
    void setUserFlags(unsigned int flags) {
        userFlags_ = flags;
    }

    /**
     * @brief Check if shader program is linkable
     *
     * @retval true if program can be linked
     * @retval false otherwise
     *
     * A program is linkable (for our purposes) it is has at least a fragment and vertex shader or
     * a compute shader.
     */
    bool isLinkable() const {
        return (hasFragment_ && hasVertex_) || (hasCompute_);
    }

    /**
     * @brief Check if shader program is linked
     *
     * @retval true if shader program is linked
     * @retval false otherwise
     */
    bool isLinked() const {
      return linked_;
    }

    /**
     * @brief Retrieve GL handle for this shader program
     *
     * @return GL handle for shader program
     */
    GLuint getHandle() const {
      return handle_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    explicit ShaderProgram(const fyusenet::GfxContextLink & context);
    void logError() const;
    void ensureExistence();
    std::vector<GLuint> getShaderHandles() const;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GLuint handle_;                                 //!< Program object handle from OpenGL
    bool bound_;                                    //!< Indicator if shader program is currently bound
    bool hasFragment_;                              //!< Set to \c true if a fragment shader is present in the shader list
    bool hasVertex_;                                //!< Set to \c true if a vertex shader is present in the shader list
    bool hasCompute_;                               //!< Set to \c true if a compute shader is present in the shader list
    bool linked_;                                   //!< Indicator if the program has been successfully linked
    unsigned int userFlags_;                        //!< Storage for user-defined flags
    std::vector<shaderptr> shaders_;                //!< Shaders which are backing the shader program
    std::unordered_map<int, GLint> symbolMap_;      //!< Mapping for symbol lookup
    mutable uint64_t hash_;                         //!< Hash code, used for content-based addressing / identity check of shader programs
};


} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
