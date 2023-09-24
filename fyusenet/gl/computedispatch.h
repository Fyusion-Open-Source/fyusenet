//--------------------------------------------------------------------------------------------------
// Project: FyuseNet
//--------------------------------------------------------------------------------------------------
// Module : Compute Shader Dispatch (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "shaderprogram.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion::opengl {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Dispatcher class for compute shaders
 *
 * This class acts as a dispatcher for compute shaders. A dispatcher controls the number of
 * shader invocations of a particular program.
 *
 * Compute shaders strongly differ from vertex or fragment shaders as they are even a bit more
 * low-level to the computational units of the GPU. As with a fragment shader, a compute shader
 * is executed on a "per item" basis, which in case of a fragment shader is a fragment or pixel.
 * In order to make use of the many shader cores in a GPU, a multitude of items (thousands) are
 * computed at the same time by different shader cores. In order to organize those items, OpenGL
 * follows the same approach as CUDA or OpenCL and organizes those in a hierarchical 3D grid.
 * The outermost layer in the hierarchy is called a "dispatch" and basically defines the program
 * that is to be executed. This dispatch can be parameterized by a 3D array of "work groups".
 * Each work group consists of a pool of "local" threads which is also parameterized as 3D array where
 * each pool runs within a single work group.
 *
 * Global work groups are executing the supplied shader program independently, which means that there
 * is no deterministic order to them. There is also no way of any synchronized computation or sharing
 * between different workgroup IDs.
 *
 * The local thread pool in each work group executes allows for a set of synchronization
 * primitives. The local pool size is controlled by the shader itself whereas the global
 * work group size is controlled by the host.
 *
 */
 // TODO (mw) finish implementation
class ComputeDispatch {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit ComputeDispatch(programptr program);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void dispatch(GLuint width, GLuint height=1, GLuint depth=1);

    /**
     * @brief Retrieve the local thread pool size
     *
     * @return Pointer to integer array with 3 elements storing the extents of each local
     *          thread pool
     */
    const GLint * localSize() const {
        return localSize_;
    }

private:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr program_;
    GLint localSize_[3] = {0, 0, 0};
};


} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
