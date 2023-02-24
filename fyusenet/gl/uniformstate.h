//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Shader Uniform State Collector (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "shaderprogram.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace opengl {

class UniformState;

typedef std::shared_ptr<UniformState> unistateptr;

/**
 * @brief Shader interface (uniforms) variable state storage class
 *
 * This class is to be used in conjunction with the ShaderProgram class and serves as a mechanism
 * to re-use shader program under varying interface conditions. It basically serves as a data-store
 * for shader variables and once applied to the target program, it will set the uniform variables
 * of the target.
 *
 *
 * @see https://www.khronos.org/opengl/wiki/Uniform_(GLSL)
 * @see ShaderProgram
 */
class UniformState {
    friend class ShaderProgram;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    UniformState(programptr target);
    ~UniformState();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    static unistateptr makeShared(programptr target) {
        return std::make_shared<UniformState>(target);
    }

    bool setUniformValue(const char *name, GLint value, bool optional=false);
    bool setUniformValue(const char *name, GLfloat value, bool optional=false);
    bool setUniformValue(GLint location, GLint value);
    bool setUniformValue(GLint location, GLfloat value);

    bool setUniformArray(const char *name, const GLfloat *v, int numEntries, bool optional=false);
    bool setUniformArray(GLint location, const GLfloat *v, int numEntries);
    bool setUniformVec2Array(const char *name, const GLfloat *v, int numEntries, bool optional=false);
    bool setUniformVec2Array(GLint location, const GLfloat *v, int numEntries);
    bool setUniformVec3Array(const char *name, const GLfloat *v, int numEntries, bool optional=false);
    bool setUniformVec3Array(GLint location, const GLfloat *v, int numEntries);

    bool setUniformVec2(const char *name, GLint v0, GLint v1, bool optional=false);
    bool setUniformVec2(const char *name, GLfloat v0, GLfloat v1, bool optional=false);
    bool setUniformVec2(GLint location, GLint v0, GLint v1);
    bool setUniformVec2(GLint location, GLfloat v0, GLfloat v1);

    bool setUniformVec3(const char *name, GLint v0, GLint v1, GLint v2, bool optional=false);
    bool setUniformVec3(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, bool optional=false);
    bool setUniformVec3(GLint location, GLint v0, GLint v1, GLint v2);
    bool setUniformVec3(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);

    bool setUniformVec4(const char *name, GLint v0, GLint v1, GLint v2, GLint v3, bool optional=false);
    bool setUniformVec4(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, bool optional=false);
    bool setUniformVec4(GLint location, GLint v0, GLint v1 ,GLint v2, GLint v3);
    bool setUniformVec4(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);

    bool setUniformMat3(const char *name, const GLfloat *matrix, bool transpose=false, bool optional=false);
    bool setUniformMat3(GLint location, const GLfloat *matrix, bool transpose) ;
    bool setUniformMat4(const char *name, const GLfloat *matrix, bool transpose=false, bool optional=false);
    bool setUniformMat4(GLint location, const GLfloat *matrix, bool transpose) ;

    void applyState(ShaderProgram *target = nullptr);

 protected:
    enum etype {
        SIGNED_INTEGER = 0,
        SIGNED_INTEGER_VEC2,
        SIGNED_INTEGER_VEC3,
        SIGNED_INTEGER_VEC4,
        FLOAT,
        FLOAT_VEC2,
        FLOAT_VEC3,
        FLOAT_VEC4,
        FLOAT_MAT3,
        FLOAT_MAT4,
        FLOAT_ARRAY,
        FLOAT_VEC2_ARRAY,
        FLOAT_VEC3_ARRAY,
        FLOAT_VEC4_ARRAY
    };
    struct entry {
        entry(etype inType,GLint loc) : type(inType), location(loc) {
        }
        etype type;
        GLint location;
        union {
            GLint i;
            GLfloat f;
            struct { GLint x,y; } ivec2;
            struct { GLint x,y,z; } ivec3;
            struct { GLint x,y,z,w; } ivec4;
            struct { GLfloat x,y; } vec2;
            struct { GLfloat x,y,z; } vec3;
            struct { GLfloat x,y,z,w; } vec4;
            struct { GLfloat mat[9]; bool transpose; } mat3;
            struct { GLfloat mat[16]; bool transpose; } mat4;
            struct { const GLfloat *values; int numEntries; } floatArray;
            struct { const GLfloat *values; int numEntries; } floatVec3Array;
        } data;
    };
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    GLint getLocation(const char *name, bool optional);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::weak_ptr<ShaderProgram> target_;       //!< Pointer to ShaderProgram (weak) that this state object refers to
    std::vector<entry> entries_;                //!< Uniform state variables
};


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
