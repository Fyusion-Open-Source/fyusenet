//--------------------------------------------------------------------------------------------------
// Project: DA Gate ISP                                                   (c) Fyusion Inc. 2021-2022
//--------------------------------------------------------------------------------------------------
// Module : Compute Shader Dispatch
// Creator: Martin Wawro
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "computedispatch.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::opengl {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Construct a dispatch environment for a compute shader
 *
 * @param program Compute shader that is to be used
 */
ComputeDispatch::ComputeDispatch(programptr program) : program_(program) {
    bool unbind = !program->isBound();
    if (!program->isBound()) program->bind();
    glGetProgramiv(program->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, localSize_);
    if (unbind) program->unbind();
}


/**
 * @brief Start/dispatch a compute shader
 *
 * @param width Width of the dispatch (global workgroup width)
 * @param height Height of the dispatch (global workgroup height)
 * @param depth Depth of the dispatch (global workgroup depth)
 */
void ComputeDispatch::dispatch(GLuint width, GLuint height, GLuint depth) {
    bool unbind = !program_->isBound();
    if (unbind) program_->bind();
    glDispatchCompute(width, height, depth);
    if (unbind) program_->unbind();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
