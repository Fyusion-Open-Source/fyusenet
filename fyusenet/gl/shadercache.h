//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Low-Level GLSL Shader Cache (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <unordered_map>
#include <memory>
#include <vector>
#include <atomic>

//-------------------------------------- Project  Headers ------------------------------------------


#include "vertexshader.h"
#include "fragmentshader.h"
#include "shaderprogram.h"
#include "../gpu/gfxcontexttracker.h"
#include "xxhash64.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace opengl {

/**
 * @brief Cache for individual shaders and shader programs
 *
 * This shader cache tried reduce strain on the GL subsystem with respect to shader memory and also
 * shader compilation time. It maintains a (global) list of instances on a per-context basis where
 * each instance is able to cache individual shaders like vertex/fragment/compute shaders and a
 * program cache, which will cache a fully compiled and linked shader program.
 *
 * To keep this cache as lightweight as possible and yet convenient enough to add some benefit,
 * we are currently using two different methods to uniquely identify a shader and a program.
 * For shaders we use a content-based method which simply computes a hash of the actual shader
 * source code and uses that to index the shader in the cache.
 *
 * For compiled and linked programs we use a different mechanism which uses the GL program handles
 * and something called a \e moduleID, which is a number that is used to modify the seed for the
 * hash computation on the handles. We use this as additional distinction based on different
 * use-cases of shader programs where the shader state might be different.
 *
 * @warning Though not likely at all, this code does not include any measures to prevent collisions
 *          on the used hashes. So, if you run into strange errors where the wrong shaders are used,
 *          please check for a hash collision.
 */
class ShaderCache : public fyusenet::GfxContextTracker {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ShaderCache(const fyusenet::GfxContextLink & link);
    virtual ~ShaderCache();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void clear();
    GLuint findShaderID(shaderptr shader) const;
    shaderptr findShader(shaderptr shader) const;
    void putShader(shaderptr shader);
    programptr findProgram(size_t moduleID, std::vector<GLuint> handles) const;
    GLuint findProgramID(size_t moduleID, std::vector<GLuint> handles) const;
    void putProgram(programptr program, size_t moduleID);
    // ------------------------------------------------------------------------
    // Static functions
    // ------------------------------------------------------------------------
    static ShaderCache * getInstance(const fyusenet::GfxContextLink & context);
    static void tearDown();
 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    uint64_t computeProgramHash(std::vector<GLuint> & handles, size_t moduleID) const;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::unordered_map<uint64_t, shaderptr> shaders_;                   //!< Cached shaders (vertex, fragment, compute)
    std::unordered_map<uint64_t, programptr> programs_;                 //!< Cached shader programs
    int seed_ = 0;                                                      //!< Seed value to compute hashes over shader content
    static std::vector<ShaderCache *> shaderCaches_;                    //!< List of shader caches (one per context)
    static std::atomic<bool> cacheLock_;                                //!< Spinlock for cache access
};

} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
