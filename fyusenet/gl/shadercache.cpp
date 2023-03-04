//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Low-Level GLSL Shader Cache
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "shadercache.h"
#ifdef FYUSENET_MULTITHREADING
#include "asyncpool.h"
#endif
#include "../common/logging.h"

//-------------------------------------- Global Variables ------------------------------------------
namespace fyusion {
namespace opengl {

//-------------------------------------- Local Definitions -----------------------------------------

std::vector<ShaderCache *> ShaderCache::shaderCaches_;
std::atomic<bool> ShaderCache::cacheLock_{false};

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param context GL context that this cache should cache programs for
 */
ShaderCache::ShaderCache(const fyusenet::GfxContextLink & context) : GfxContextTracker(), seed_(0) {
    setContext(context);
    assertContext();
}

/**
 * @brief Destructor
 *
 * @pre Must be called with the cache GL context being the current one in case clear() has not
 *      been called before the destructor.
 *
 * Removes cached resources from main memory and GL subsystem if required. A previous call to
 * clear() will prevent this destructor from cleaning up GL resources.
 *
 * @see clear()
 */
ShaderCache::~ShaderCache() {
    if ((!programs_.empty()) || (!shaders_.empty())) clear();
}


/**
 * @brief Clears cached resources from system memory and GL subsystem
 *
 * @pre Must be called with the cache GL context being the current one
 *
 * Removes cached resources from main memory and GL subsystem.
 */
void ShaderCache::clear() {
    assertContext();
    programs_.clear();
    shaders_.clear();
}


/**
 * @brief Find shader program in cache
 *
 * @param moduleID Identifier for the module/class that is querying
 * @param handles Vector of shader handles that we want to find a cached (linked) program for
 *
 * @return Shared pointer to ShaderProgram object that fulfills query, or empty pointer if not found.
 *
 * Returns a shared ShaderProgram object that meets the query criteria, i.e. the program was
 * created from the supplied shader handles and was created under the same \p moduleID as the
 * one supplied in the query. If not such object is found, an empty (shared) object is returned.
 *
 * @note This function is not thread-safe, however we assume that it is only called from within
 *       the thread which is associated to the pertaining OpenGL context, thus we do not need
 *       to make it thread-safe.
 */
programptr ShaderCache::findProgram(size_t moduleID, std::vector<GLuint> handles) const {
    if (handles.empty()) return 0;
    uint64_t hash = computeProgramHash(handles,moduleID);
    auto it = programs_.find(hash);
    if (it == programs_.end()) return programptr();
    else return it->second;
}


/**
 * @brief Find shader program GL handle in cache
 *
 * @param moduleID Identifier for the module/class that is querying
 * @param handles Vector of shader handles that we want to find a cached (linked) program for
 *
 * @return GL handle of shader program or 0 if it was not found
 *
 * Returns a GL handle of a shader program that meets the query criteria, i.e. the program was
 * created from the supplied shader handles and was created under the same \p moduleID as the
 * one supplied in the query. If not such object is found, a zero handle is returned.
 *
 * @note This function is not thread-safe, however we assume that it is only called from within
 *       the thread which is associated to the pertaining OpenGL context, thus we do not need
 *       to make it thread-safe.
 */
GLuint ShaderCache::findProgramID(size_t moduleID, std::vector<GLuint> handles) const {
    programptr ptr = findProgram(moduleID, handles);
    if (ptr.get()) {
        return ptr.get()->handle_;
    } else return 0;
}


/**
 * @brief Put a compiled and linked shader program into the shader cache
 *
 * @param program Shared pointer to linked GL program
 * @param moduleID Identifier for the module/class that the program was created under
 *
 * @note This function is not thread-safe, however we assume that it is only called from within
 *       the thread which is associated to the pertaining OpenGL context, thus we do not need
 *       to make it thread-safe.
 */
void ShaderCache::putProgram(programptr program, size_t moduleID) {
    std::vector<GLuint> handles = program->getShaderHandles();
    if (handles.empty()) THROW_EXCEPTION_ARGS(GLException,"Cannot add program to cache, no shader handles found");
    uint64_t hash = computeProgramHash(handles, moduleID);
    program->hash_ = hash;
    programs_[hash] = program;
}


/**
 * @brief Query shader (not shader program) from cache
 *
 * @param shader Shader pointer that contains the code for the shader to query
 *
 * @return Shared pointer to shader in the cache, or empty pointer if no such shader was found.
 *
 * Query shader cache for an existing vertex/fragment/compute shader that matches the supplied
 * query \p shader.
 *
 * Though it might seen counterintuitive, this function queries a shader in the cache using a
 * shader. The reason for this is that there is not need to compile the supplied shader, it is
 * sufficient to set the code into the \p shader and then query the cache, which will do the
 * content-based addressing.
 */
shaderptr ShaderCache::findShader(shaderptr shader) const {
    uint64_t hash = XXHash64::hash(shader->getCode(),seed_);
    auto it = shaders_.find(hash);
    if (it == shaders_.end()) return shaderptr();
    else {
        return it->second;
    }
}



/**
 * @brief Query shader GL handle (not shader program) from cache
 *
 * @param shader Shader pointer that contains the code for the shader to query
 *
 * @return GL handle that is associated with \p shader or 0 if not such shader was found in the
 *         cache.
 *
 * Query shader cache for an existing vertex/fragment/compute shader that matches the supplied
 * query \p shader.
 *
 * Though it might seen counterintuitive, this function queries a shader in the cache using a
 * shader. The reason for this is that there is no need to compile the supplied shader, it is
 * sufficient to set the code into the \p shader and then query the cache, which will do the
 * content-based addressing. If no matching shader was found, the supplied \p shader can just
 * be compiled.
 */
GLuint ShaderCache::findShaderID(shaderptr shader) const {
    uint64_t hash = XXHash64::hash(shader->getCode(),seed_);
    auto it = shaders_.find(hash);
    if (it == shaders_.end()) return 0;
    else {
        return it->second->handle_;
    }
}


/**
 * @brief Put a compiled vertex/fragment/compute shader into the cache
 *
 * @param shader Shared pointer to a compiled vertex/fragment/compute shader which should be cached
 *
 * @pre The supplied \p shader must have been (successfully) compiled before putting it into the
 *      cache
 */
void ShaderCache::putShader(shaderptr shader) {
    if (!shader->isCompiled()) THROW_EXCEPTION_ARGS(GLException, "Shader must be compiled before being put into the cache");
    uint64_t hash = XXHash64::hash(shader->getCode(),seed_);
    shader->hash_ = hash;
    shaders_[hash] = shader;
}



/**
 * @brief Perform cleanup of shader caches
 *
 * @pre The AsyncPool is still in operational state and not shut down already
 *
 * @post Caches are cleared (including GL resources)
 *
 * Use this function at the end of program execution to make sure that all GL resources occupied
 * by the cache are released.
 */
void ShaderCache::tearDown() {
    bool expt = false;
    while (!cacheLock_.compare_exchange_strong(expt,true)) {
        expt = false;
    }
    for (int i=0; i < (int)shaderCaches_.size(); i++) {
        if (shaderCaches_[i]) {
            ShaderCache * cache = shaderCaches_[i];
            shaderCaches_[i] = nullptr;
            if (cache->context().isCurrent()) {
                cache->clear();
            } else {
#ifdef FYUSENET_MULTITHREADING
                AsyncPool::GLThread thread = AsyncPool::getContextThread(cache->context());
                assert(thread.isValid());
                thread->waitTask([&]() { cache->clear(); });
#else
                assert(false);
                // TODO (mw) throw an exception here instead ?
#endif
            }
            delete cache;
        }
    }
    cacheLock_.store(false);
}


/**
 * @brief Retrieve shader cache instance for specified context
 *
 * @param ctx GL context to get shader cache for
 *
 * @return Pointer to a shader cache instance. If a cache was not present for the context,
 *         a new cache will be created.
 *
 * @warning Do not store the pointer, treat it as transient.
 *
 * @todo Use a shared pointer instead ?
 */
ShaderCache * ShaderCache::getInstance(const fyusenet::GfxContextLink & ctx) {
    bool expt = false;
    while (!cacheLock_.compare_exchange_strong(expt,true)) {
        expt=false;
    }
    for (int i=0; i < (int)shaderCaches_.size(); i++) {
        if (shaderCaches_[i] && shaderCaches_[i]->context_ == ctx) {
            // NOTE (mw) if tearDown() is called on a different thread, this pointer will become invalid
            ShaderCache * cache = shaderCaches_[i];
            cacheLock_.store(false);
            return cache;
        }
    }
    ShaderCache * cache = new ShaderCache(ctx);
    shaderCaches_.push_back(cache);
    cacheLock_.store(false);
    return cache;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Compute 64-bit hash for a set of shader handles and a module ID
 *
 * @param handles Vector of shader handles (not program handles) that make up a GL program
 * @param moduleID Identifier for the module/class that the program was created under
 *
 * @return 64-bit hash value that can be used as hash-value for a shader program
 *
 * This function computes a hash based on the supplied \p moduleID and the GL handles, which are
 * simply sorted and then fed into a hash computation.
 *
 * @warning The \p handles will be sorted in place and are subject to change their order
 */
uint64_t ShaderCache::computeProgramHash(std::vector<GLuint>& handles, size_t moduleID) const {
    std::sort(handles.begin(),handles.end(),[](GLuint t1,GLuint t2) { return  (t1 < t2); });
    uint64_t hash = XXHash64::hash((const void *)(&handles[0]),sizeof(GLuint)*handles.size(),seed_ + moduleID);
    return hash;
}

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
