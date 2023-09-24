//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Vertex Array Object (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "glexception.h"
#include "../gpu/gfxcontextlink.h"
#include "../common/logging.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace opengl {

/**
 * @brief Vertex array object wrapper
 *
 * This class wraps a vertex array object, which basically stores the state of associated buffer
 * objects like VBOs or IBOs.
 *
 * @see https://www.khronos.org/opengl/wiki/Vertex_Specification#Vertex_Array_Object
 */
class VAO {
 public:
    /**
     * @brief Constructor
     *
     * @param context Link to GL context that the VAO runs in
     */
    VAO(const fyusenet::GfxContextLink & context = fyusenet::GfxContextLink()) : context_(context) {
        glGenVertexArrays(1,&handle_);
        if (handle_ == 0) {
            THROW_EXCEPTION_ARGS(GLException,"Cannot create VAO (err=0x%X)",glGetError());
        }
        bound_=false;
    }

    /**
     * @brief Destructor
     *
     * @pre GL context that is stored in this instance is current to the calling thread
     *
     * Frees GL resources (not the buffers that are part of the %VAO, just the %VAO itself)
     */
    ~VAO() {
        if (context_.isCurrent()) {
            if (handle_ != 0) {
                if (bound_) unbind();
                glDeleteVertexArrays(1,&handle_);
            }
        } else {
            FNLOGE("Trying to destroy VAO from wrong GL context");
        }
        handle_ = 0;
    }

    /**
     * @brief Bind VAO and its associated buffers
     *
     * @retval true bind operation was successful
     * @retval false otherwise
     */
    bool bind() {
#ifdef DEBUG
        if (!context_.isCurrent()) {
            FNLOGE("Trying to use VAO from wrong GL context");
            return false;
        }
#endif
        glBindVertexArray(handle_);
        bound_ = true;
        return true;
    }

    /**
     * @brief Release %VAO binding
     */
    void unbind() {
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER,0);
        bound_ = false;
    }

    /**
     * @brief Enable vertex attribute array with given index
     *
     * @param arrIndex Array index to enable
     */
    void enableArray(int arrIndex) {
        glEnableVertexAttribArray(arrIndex);
    }

    /**
     * @brief Disable vertex attribute array with given index
     *
     * @param arrIndex Array index to disable
     */
    void disableArray(int arrIndex) {
        glDisableVertexAttribArray(arrIndex);
    }

    /**
     * @brief Set floating-point vertex attributes for specified vertex array attribute buffer for currently bound array buffer
     *
     * @param index Index of the vertex attribute buffer
     * @param components Number of components per attribute (1..4)
     * @param type Data type of the components for the array (e.g. \c GL_FLOAT, or \c GL_HALF_FLOAT)
     * @param normalized Indicator if data is normalized
     * @param stride Byte offset between consecutive attributes, 0 indicates that they are contiguous
     * @param offset Offset (in bytes) the data starts on in the currently bound \c GL_ARRAY_BUFFER target
     *
     * @see enableArray()
     */
    void setVertexAttributeBuffer(GLuint index, GLint components, GLenum type, GLboolean normalized, GLsizei stride, unsigned int offset) {
#ifdef DEBUG
        switch (type) {
            case GL_INT:
            case GL_UNSIGNED_INT:
            case GL_UNSIGNED_SHORT:
            case GL_SHORT:
            case GL_UNSIGNED_BYTE:
            case GL_BYTE:
                THROW_EXCEPTION_ARGS(GLException,"Trying to provide integer data to floating-point attribute buffer");
            default:
              break;
        }
        glGetError();
#endif
        glVertexAttribPointer(index,components,type,normalized,stride,(const char *)0+offset);
#ifdef DEBUG
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Set vertex attribute pointer for index %d failed (glerr=0x%X)",index,err);
#endif
    }


    /**
     * @brief Set integer vertex attributes for specified vertex array attribute buffer for currently bound array buffer
     *
     * @param index Index of the vertex attribute buffer
     * @param components Number of components per attribute (1..4)
     * @param type Data type of the components for the array (e.g. \c GL_INT, or \c GL_BYTE)
     * @param stride Byte offset between consecutive attributes, 0 indicates that they are contiguous
     * @param offset Offset (in bytes) the data starts on in the currently bound \c GL_ARRAY_BUFFER target
     */
    void setVertexAttributeBuffer(GLuint index,GLint components,GLenum type,GLsizei stride,unsigned int offset) {
#ifdef DEBUG
        switch (type) {
            case GL_FLOAT:
            case GL_HALF_FLOAT:
                THROW_EXCEPTION_ARGS(GLException,"Trying to provide floating-point data to integer attribute buffer");
            default:
                break;
        }
#endif
        glGetError();
        glVertexAttribIPointer(index,components,type,stride,(const char *)0+offset);
#ifdef DEBUG
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Set vertex attribute pointer for index %d failed (glerr=0x%X)",index,err);
#endif
    }

    /**
     * @brief Check if VAO is valid
     *
     * @retval true VAO is valid
     * @retval false VAO is invalid
     */
    bool isValid() const {
        return (glIsVertexArray(handle_) == GL_TRUE);
    }

 private:
    GLuint handle_ = 0;                     //!< Raw GL handle for the %VAO
    fyusenet::GfxContextLink context_;      //!< Context the %VAO runs in
    bool bound_ = false;                    //!< Indicator if %VAO is bound
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
