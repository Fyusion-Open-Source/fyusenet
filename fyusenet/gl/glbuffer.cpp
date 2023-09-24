//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Generic OpenGL Buffer Object
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "glbuffer.h"
#include "glexception.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion::opengl {


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param target Default target to bind this buffer to
 * @param context OpenGL context that this buffer is created under (must be the current context)
 *
 * @see genBuffer()
 */
GLBuffer::GLBuffer(GLenum target,const fyusenet::GfxContextLink &context) : GfxContextTracker() ,target_(target),handle_(0),bound_(false) {
    setContext(context);
    genBuffer();
}

/**
 * @brief Constructor for an object around an existing handle
 *
 * @param target Default target to bind this buffer to
 * @param handle Handle to wrap with this object
 * @param bound Indicator if buffer object is currently bound
 * @param context OpenGL context that this buffer is created under
 */
GLBuffer::GLBuffer(GLenum target,GLuint handle,bool bound,const fyusenet::GfxContextLink &context) : GfxContextTracker(),target_(target),handle_(handle),bound_(bound) {
    setContext(context);
}


/**
 * @brief Destructor
 *
 * Deletes the buffer object from GL resources.
 *
 * @pre Calling thread has the OpenGL context of this buffer or a context that is shared with that
 */
GLBuffer::~GLBuffer() {
    if (handle_ != 0) {
        if (bound_) unbind();
        glDeleteBuffers(1,&handle_);
    }
    handle_ = 0;
}


/**
 * @brief Bind buffer object to default target
 *
 * @pre Calling thread has the OpenGL context of this buffer or a context that is shared with that
 * @post GL handle wrapped by this object will be bound to the default buffer target
 *
 * @throws GLException on GL errors for debug builds
 */
void GLBuffer::bind() {
#ifdef DEBUG    
    // NOTE (mw) we do not assert a specific context here, since buffer objects may be shared
    // between (shared) contexts and we assume that we only pass those buffers between shared
    // contexts
    if (bound_) {
        FNLOGW("Buffer was already bound, please check your code if you missed an unbind");
    }
    glGetError();
#endif
    bind(target_);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Buffer %d binding to %X failed (glerr=0x%X)",handle_,target_,err);
#endif
}


/**
 * @brief Bind buffer object to specific target
 *
 * @param target GL target to bind buffer object to
 *
 * @pre Calling thread has the OpenGL context of this buffer or a context that is shared with that
 * @post GL handle wrapped by this object will be bound to supplied buffer \p target
 *
 * @throws GLException on GL errors for debug builds
 */
void GLBuffer::bind(GLenum target) {
#ifdef DEBUG
    // NOTE (mw) we do not assert a specific context here, since buffer objects may be shared
    // between (shared) contexts and we assume that we only pass those buffers between shared
    // contexts
    glGetError();
    if (bound_) {
        FNLOGW("Binding buffer to 0x%X though it was already bound, check your code for missing unbinds", target);
    }
#endif
    if (handle_ == 0) THROW_EXCEPTION_ARGS(GLException,"Trying to bind uninitialized buffer");
    glBindBuffer(target, handle_);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Buffer %d binding to %X failed (glerr=0x%X)",handle_,target,err);
#endif
    bound_ = true;
}


/**
 * @brief Unbinds buffer object from default target
 *
 * @pre Calling thread has the OpenGL context of this buffer or a context that is shared with that
 */
void GLBuffer::unbind() {
    unbind(target_);
}


/**
 * @brief Unbinds buffer object from specific target
 *
 * @pre Calling thread has the OpenGL context of this buffer or a context that is shared with that
 * @param target GL target to unbind buffer object from
 */
void GLBuffer::unbind(GLenum target) {
    glBindBuffer(target,0);
    bound_ = false;
}


/**
 * @brief Write data to buffer object
 *
 * @param data Pointer to data that should be written to the buffer object, \c nullptr will clear
 *             the buffer
 * @param dataSize Number of bytes to write into the buffer
 * @param usage Specifies data usage, e.g. \c GL_STREAM_DRAW or \c GL_STATIC_READ etc.
 *
 * @pre Calling thread has the OpenGL context of this buffer or a context that is shared with that
 *
 * @throws GLException on GL errors for debug builds
 *
 * @see https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBufferData.xhtml
 */
void GLBuffer::setBufferData(void *data, int dataSize, GLenum usage) {
    if (!bound_) bind();
#ifdef DEBUG
    glGetError();
#endif
    glBufferData(target_, dataSize, data, usage);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(GLException,"Cannot set buffer data for buffer %d target 0x%X (glerr=0x%X)",handle_,target_,err);
#endif
    unbind();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Generate buffer handle
 *
 * @throws GLException if handle could not be generated
 */
void GLBuffer::genBuffer() {
    if (handle_ != 0) glDeleteBuffers(1, &handle_);
    handle_ = 0;
    glGenBuffers(1, &handle_);
    if (handle_ == 0) THROW_EXCEPTION_ARGS(GLException,"Cannot generate buffer object handle");
}

} // fyusion::opengl namespace

// vim: set expandtab ts=4 sw=4:
