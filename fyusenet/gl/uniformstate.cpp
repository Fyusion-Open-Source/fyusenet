//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Shader Uniform State Collector
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "glexception.h"
#include "shaderprogram.h"
#include "uniformstate.h"
#include "shaderexception.h"

//-------------------------------------- Global Variables ------------------------------------------
namespace fyusion {
namespace opengl {


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param target Shared pointer to ShaderProgram instance that this state is decorating
 *
 * Creates a UniformState object around the supplied shader program. The supplied \p target is
 * stored as a weak pointer in this object.
 */
UniformState::UniformState(programptr target) : target_(target) {
}


/**
 * @brief Destructor (idle)
 */
UniformState::~UniformState() {
}



/**
 * @brief Set a uniform variable by ist name (single 32-bit integer)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param value Value to set to the uniform
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied value in this state object and associates it with the uniform
 * variable in the shader that has the matching name. It does not set the variable in the actual
 * shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformValue(const char *name, GLint value, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformValue(loc, value);
}



/**
 * @brief Set a uniform variable by ist name (single float)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param value Value to set to the uniform
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied value in this state object and associates it with the uniform
 * variable in the shader that has the matching name. It does not set the variable in the actual
 * shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformValue(const char *name, GLfloat value, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformValue(loc, value);
}


/**
 * @brief Set a uniform variable by its location (single 32-bit integer)
 *
 * @param location Location ID in GL shader
 *
 * @param value Value to set to the uniform
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied value in this state object and associates it with the uniform
 * variable in the shader that has the matching .ocation. It does not set the variable in the actual
 * shader at this point in time.
 */
bool UniformState::setUniformValue(GLint location, GLint value)  {
    if (location < 0) return false;
    entry ent(SIGNED_INTEGER, location);
    ent.data.i = value;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform variable by its location (single float)
 *
 * @param location Location ID in GL shader
 *
 * @param value Value to set to the uniform
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied value in this state object and associates it with the uniform
 * variable in the shader that has the matching .ocation. It does not set the variable in the actual
 * shader at this point in time.
 */
bool UniformState::setUniformValue(GLint location, GLfloat value)  {
    if (location < 0) return false;
    entry ent(FLOAT, location);
    ent.data.f = value;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform float array by name
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit floating-point data which should be set to the uniform array
 *
 * @param numEntries Number of floating-point values in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching name. It does not set the variable
 * in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false
 *
 * @warning The supplied \p data is \b not deep-copied, do not overwrite or delete the pointer
 *          as long as this object is not deleted.
 *
 * @todo Deep-copy data.
 */
bool UniformState::setUniformArray(const char *name, const GLfloat *data, int numEntries, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformArray(loc, data, numEntries);
}


/**
 * @brief Set a uniform float array by GLSL location
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param numEntries Number of floats in the \p data array
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @warning The supplied \p data is \b not deep-copied, do not overwrite or delete the pointer
 *          as long as this object is not deleted.
 *
 * @todo Deep-copy data.
 */
bool UniformState::setUniformArray(GLint location, const GLfloat *data, int numEntries) {
    if (location < 0) return false;
    entry ent(FLOAT_ARRAY, location);
    // TODO (mw) a deep-copy would be better here
    ent.data.floatArray.values = data;
    ent.data.floatArray.numEntries = numEntries;
    entries_.push_back(ent);
    return true;
}

/**
 * @brief Set a uniform vec2 float array by name
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit floating-point data which should be set to the uniform array
 *
 * @param num2Entries Number of vec2 floating-point values in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching name. It does not set the variable
 * in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false
 *
 * @warning The supplied \p data is \b not deep-copied, do not overwrite or delete the pointer
 *          as long as this object is not deleted.
 *
 * @todo Deep-copy data.
 */
bool UniformState::setUniformVec2Array(const char *name, const GLfloat *data, int num2Entries, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformVec2Array(loc, data, num2Entries);
}



/**
 * @brief Set a uniform vec2 float array by GLSL location
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param num2Entries Number of vec2 floating-point values in the \p data array
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @warning The supplied \p data is \b not deep-copied, do not overwrite or delete the pointer
 *          as long as this object is not deleted.
 *
 * @todo Deep-copy data.
 */
bool UniformState::setUniformVec2Array(GLint location, const GLfloat *data, int num2Entries) {
    if (location < 0) return false;
    entry ent(FLOAT_VEC2_ARRAY, location);
    // TODO (mw) a deep-copy would be better here
    ent.data.floatArray.values = data;
    ent.data.floatArray.numEntries = num2Entries;
    entries_.push_back(ent);
    return true;
}

/**
 * @brief Set a uniform vec3 float array by name
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit floating-point data which should be set to the uniform array
 *
 * @param num3Entries Number of vec3 floating-point values in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching name. It does not set the variable
 * in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false
 *
 * @warning The supplied \p data is \b not deep-copied, do not overwrite or delete the pointer
 *          as long as this object is not deleted.
 *
 * @todo Deep-copy data.
 */
bool UniformState::setUniformVec3Array(const char *name, const GLfloat *data, int num3Entries, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformVec3Array(loc, data, num3Entries);
}



/**
 * @brief Set a uniform vec3 float array by GLSL location
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param num3Entries Number of vec3 floating-point values in the \p data array
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @warning The supplied \p data is \b not deep-copied, do not overwrite or delete the pointer
 *          as long as this object is not deleted.
 *
 * @todo Deep-copy data.
 */
bool UniformState::setUniformVec3Array(GLint location, const GLfloat *data, int num3Entries) {
    if (location < 0) return false;
    entry ent(FLOAT_VEC3_ARRAY,location);
    // TODO (mw) a deep-copy would be better here
    ent.data.floatArray.values = data;
    ent.data.floatArray.numEntries = num3Entries;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform variable by its name (32-bit integer ivec2)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of ivec2 to set
 * @param v1 2nd component of ivec2 to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates them with the
 * uniform variable in the shader that has the matching name. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformVec2(const char *name, GLint v0, GLint v1, bool optional)  {
    GLint loc = getLocation(name,optional);
    return setUniformVec2(loc,v0,v1);
}


/**
 * @brief Set a uniform variable by its name (floating-point vec2)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of vec2 to set
 * @param v1 2nd component of vec2 to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates them with the
 * uniform variable in the shader that has the matching name. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformVec2(const char *name, GLfloat v0, GLfloat v1, bool optional)  {
    GLint loc = getLocation(name,optional);
    return setUniformVec2(loc,v0,v1);
}


/**
 * @brief Set a uniform variable by its location ID (32-bit integer ivec2)
 *
 * @param location Location ID in GL shader
 *
 * @param v0 1st component of ivec2 to set
 * @param v1 2nd component of ivec2 to set
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 */
bool UniformState::setUniformVec2(GLint location, GLint v0, GLint v1) {
    if (location < 0) return false;
    entry ent(SIGNED_INTEGER_VEC2,location);
    ent.data.ivec2.x=v0;
    ent.data.ivec2.y=v1;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform variable by its location ID (floating-point vec2)
 *
 * @param location Location ID in GL shader
 *
 * @param v0 1st component of vec2 to set
 * @param v1 2nd component of vec2 to set
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 */
bool UniformState::setUniformVec2(GLint location, GLfloat v0, GLfloat v1) {
    if (location < 0) return false;
    entry ent(FLOAT_VEC2,location);
    ent.data.vec2.x=v0;
    ent.data.vec2.y=v1;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform variable by its name (32-bit integer ivec3)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of ivec3 to set
 * @param v1 2nd component of ivec3 to set
 * @param v2 3rd component of ivec3 to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates them with the
 * uniform variable in the shader that has the matching name. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformVec3(const char *name, GLint v0, GLint v1, GLint v2, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformVec3(loc, v0, v1, v2);
}


/**
 * @brief Set a uniform variable by its name (floating-point vec3)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of vec3 to set
 * @param v1 2nd component of vec3 to set
 * @param v2 3rd component of vec3 to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates them with the
 * uniform variable in the shader that has the matching name. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformVec3(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformVec3(loc, v0, v1, v2);
}


/**
 * @brief Set a uniform variable by its location ID (32-bit integer ivec3)
 *
 * @param location Location ID in GL shader
 *
 * @param v0 1st component of ivec3 to set
 * @param v1 2nd component of ivec3 to set
 * @param v2 3rd component of ivec3 to set
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 */
bool UniformState::setUniformVec3(GLint location, GLint v0, GLint v1, GLint v2) {
    if (location < 0) return false;
    entry ent(SIGNED_INTEGER_VEC3, location);
    ent.data.ivec3.x = v0;
    ent.data.ivec3.y = v1;
    ent.data.ivec3.z = v2;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform variable by its location ID (floating-point vec3)
 *
 * @param location Location ID in GL shader
 *
 * @param v0 1st component of vec3 to set
 * @param v1 2nd component of vec3 to set
 * @param v2 3rd component of vec3 to set
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the contents of the supplied array in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 */
bool UniformState::setUniformVec3(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    if (location < 0) return false;
    entry ent(FLOAT_VEC3,location);
    ent.data.vec3.x = v0;
    ent.data.vec3.y = v1;
    ent.data.vec3.z = v2;
    entries_.push_back(ent);
    return true;
}



/**
 * @brief Set a uniform variable by its name (32-bit integer ivec4)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of ivec4 to set
 * @param v1 2nd component of ivec4 to set
 * @param v2 3rd component of ivec4 to set
 * @param v3 4th component of ivec4 to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates them with the
 * uniform variable in the shader that has the matching name. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformVec4(const char *name, GLint v0, GLint v1, GLint v2, GLint v3, bool optional) {
    GLint loc = getLocation(name,optional);
    return setUniformVec4(loc, v0, v1, v2, v3);
}


/**
 * @brief Set a uniform variable by its name (floating-point vec4)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of vec4 to set
 * @param v1 2nd component of vec4 to set
 * @param v2 3rd component of vec4 to set
 * @param v3 4th component of vec4 to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates them with the
 * uniform variable in the shader that has the matching name. It does not set the
 * variable in the actual shader at this point in time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 */
bool UniformState::setUniformVec4(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, bool optional) {
    GLint loc = getLocation(name, optional);
    return setUniformVec4(loc, v0, v1, v2, v3);
}


/**
 * @brief Set a uniform variable by its location ID (32-bit integer ivec4)
 *
 * @param location Location ID in GL shader
 *
 * @param v0 1st component of ivec4 to set
 * @param v1 2nd component of ivec4 to set
 * @param v2 3rd component of ivec4 to set
 * @param v3 4th component of ivec4 to set
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied values in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 */
bool UniformState::setUniformVec4(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    if (location < 0) return false;
    entry ent(SIGNED_INTEGER_VEC4, location);
    ent.data.ivec4.x = v0;
    ent.data.ivec4.y = v1;
    ent.data.ivec4.z = v2;
    ent.data.ivec4.w = v3;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform variable by its location ID (floating-point vec4)
 *
 * @param location Location ID in GL shader
 *
 * @param v0 1st component of vec4 to set
 * @param v1 2nd component of vec4 to set
 * @param v2 3rd component of vec4 to set
 * @param v3 4th component of vec4 to set
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied parameters in this state object and associates it
 * with the uniform variable in the shader that has the matching location. It does not set the
 * variable in the actual shader at this point in time.
 */
bool UniformState::setUniformVec4(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)  {
    if (location < 0) return false;
    entry ent(FLOAT_VEC4, location);
    ent.data.vec4.x = v0;
    ent.data.vec4.y = v1;
    ent.data.vec4.z = v2;
    ent.data.vec4.w = v3;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform matrix by its name (floating-point 3x3 matrix)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param matrix Pointer to float matrix data (9 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied matrix components in this object and associates it with the
 * uniform variable referenced to by the \p name. It does not set the variable at this point in
 * time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 *
 * @note The supplied \p matrix is deep-copied, no need to retain it after using this function.
 */
bool UniformState::setUniformMat3(const char *name, const GLfloat *matrix, bool transpose, bool optional)  {
    GLint loc = getLocation(name, optional);
    return setUniformMat3(loc, matrix, transpose);
}



/**
 * @brief Set a uniform matrix by its location (floating-point 3x3 matrix)
 *
 * @param location Location ID in GL shader
 *
 * @param matrix Pointer to float matrix data (9 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied matrix components in this object and associates it with the
 * uniform variable referenced to by the \p location. It does not set the variable at this point in
 * time.
 *
 * @note The supplied \p matrix is deep-copied, no need to retain it after using this function.
 */
bool UniformState::setUniformMat3(GLint location, const GLfloat *matrix, bool transpose)  {
    if (location < 0) return false;
    entry ent(FLOAT_MAT3, location);
    memcpy(ent.data.mat3.mat, matrix, 3 * 3 * sizeof(GLfloat));
    ent.data.mat3.transpose = transpose;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Set a uniform matrix by its name (floating-point 4x4 matrix)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param matrix Pointer to float matrix data (16 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied matrix components in this object and associates it with the
 * uniform variable referenced to by the \p name. It does not set the variable at this point in
 * time.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false
 *
 * @note The supplied \p matrix is deep-copied, no need to retain it after using this function.
 */
bool UniformState::setUniformMat4(const char *name, const GLfloat *matrix, bool transpose, bool optional)  {
    GLint loc = getLocation(name, optional);
    return setUniformMat4(loc, matrix, transpose);
}

/**
 * @brief Set a uniform matrix by its location (floating-point 4x4 matrix)
 *
 * @param location Location ID in GL shader
 *
 * @param matrix Pointer to float matrix data (9 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @retval true if value was stored
 * @retval false if value was not stored
 *
 * This function stores the supplied matrix components in this object and associates it with the
 * uniform variable referenced to by the \p location. It does not set the variable at this point in
 * time.
 *
 * @note The supplied \p matrix is deep-copied, no need to retain it after using this function.
 */
bool UniformState::setUniformMat4(GLint location, const GLfloat *matrix, bool transpose)  {
    if (location < 0) return false;
    entry ent(FLOAT_MAT4, location);
    memcpy(ent.data.mat4.mat, matrix, 4 * 4 * sizeof(GLfloat));
    ent.data.mat4.transpose = transpose;
    entries_.push_back(ent);
    return true;
}


/**
 * @brief Apply stored state to the underlying shader program
 *
 * @param target Optional pointer to shader program (for verification) that the state shall be
 *               applied to
 *
 * @pre The target shader program must be bound / active
 *
 * This function applies the state stored in this object to the shader program it wraps. The
 * \p target parameter is merely there for verification, which means that an exception will
 * be thrown if the \p target it not a \c nullptr and also not identical to the shader program
 * that was wrapped by this state object.
 *
 * This function uses the uniform interface of the ShaderProgram instance itself to set the
 * variables into the shader.
 *
 * @see #target_
 *
 * @throws ShaderException in case the \p target is not a \c nullptr and not equal to the wrapped
 *                         program or in case the shader program was not active.
 */
void UniformState::applyState(ShaderProgram *target) {
    auto ptr = target_.lock();
    if (!ptr) return;
    if ((target) && (ptr.get() != target)) THROW_EXCEPTION_ARGS(ShaderException,"Cannot apply state to shader it was not created for");
    for (const entry& ent : entries_) {
        switch (ent.type) {
            case SIGNED_INTEGER:
                ptr->setUniformValue(ent.location, ent.data.i);
                break;
            case SIGNED_INTEGER_VEC2:
                ptr->setUniformVec2(ent.location, ent.data.ivec2.x, ent.data.ivec2.y);
                break;
            case SIGNED_INTEGER_VEC3:
                ptr->setUniformVec3(ent.location, ent.data.ivec3.x, ent.data.ivec3.y, ent.data.ivec3.z);
                break;
            case SIGNED_INTEGER_VEC4:
                ptr->setUniformVec4(ent.location, ent.data.ivec4.x, ent.data.ivec4.y, ent.data.ivec4.z, ent.data.ivec4.w);
                break;
            case FLOAT:
                ptr->setUniformValue(ent.location, ent.data.f);
                break;
            case FLOAT_VEC2:
                ptr->setUniformVec2(ent.location, ent.data.vec2.x, ent.data.vec2.y);
                break;
            case FLOAT_VEC3:
                ptr->setUniformVec3(ent.location, ent.data.vec3.x, ent.data.vec3.y, ent.data.vec3.z);
                break;
            case FLOAT_VEC4:
                ptr->setUniformVec4(ent.location, ent.data.vec4.x, ent.data.vec4.y, ent.data.vec4.z, ent.data.vec4.w);
                break;
            case FLOAT_MAT3:
                ptr->setUniformMat3(ent.location, ent.data.mat3.mat, ent.data.mat3.transpose);
                break;
            case FLOAT_MAT4:
                ptr->setUniformMat4(ent.location, ent.data.mat4.mat, ent.data.mat4.transpose);
                break;
            case FLOAT_ARRAY:
                ptr->setUniformArray(ent.location, ent.data.floatArray.values, ent.data.floatArray.numEntries);
                break;
            case FLOAT_VEC2_ARRAY:
                ptr->setUniformVec2Array(ent.location, ent.data.floatArray.values, ent.data.floatArray.numEntries);
                break;
            case FLOAT_VEC3_ARRAY:
                ptr->setUniformVec3Array(ent.location, ent.data.floatArray.values, ent.data.floatArray.numEntries);
                break;
            default:
                // not implemented (yet) or illegal
                assert(false);
        }
    }
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Resolve uniform name to location ID
 *
 * @param name Name of the uniform to look up
 *
 * @param optional When set to \c true, this function silently returns a -1 if the uniform could
 *                 not be resolved. Otherwise it will throw an exception.
 *
 * @return GLSL location ID or -1 if location was not found
 *
 * @throws ShaderException in case \p optional was set to \c false.
 */
GLint UniformState::getLocation(const char *name, bool optional) {
    auto ptr = target_.lock();
    if (!ptr) THROW_EXCEPTION_ARGS(ShaderException, "No shader supplied or expired");
    return ptr->resolveLocation(name, optional);
}


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
