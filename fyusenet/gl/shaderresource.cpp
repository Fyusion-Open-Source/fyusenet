//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Resource (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "shaderresource.h"
#include "shadercache.h"
#include "vertexshader.h"
#include "fragmentshader.h"

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
 * @param shader Pointer to string with shader source-coe
 * @param resourceName Name within the resource system to represent the shader source by
 *
 * This constructor registers the supplied shader source code under the specified resource
 * name with the ShaderRepository singleton. The resulting object may be destroyed immediately
 * as it does not take ownership over the source code or the name string.
 */
ShaderResource::ShaderResource(const char *shader, const char *resourceName) {
    ShaderRepository & repo = ShaderRepository::repository();
    repo.addResource(shader,resourceName);
}


/**
 * @brief Retrieve shader source by its resource name
 *
 * @param resourceName Name of resource (in virtual filesystem) to retrieve
 *
 * @return Pointer to shaer source, or \c nullptr if no such shader exists
 */
const char * ShaderRepository::getShader(const char *resourceName) {
    if (!resourceName) return nullptr;
    ShaderRepository & repo = repository();
    auto const & it = repo.shaderMap_.find(std::string(resourceName));
    if (it == repo.shaderMap_.end()) return nullptr;
    else return it->second;
}

/**
 * @brief Preprocess and compile/cache a vertex/fragment shader pair
 *
 * @param vertexName Resource name for the vertex shader to use
 *
 * @param fragmentName Resource name for the fragment shader to use
 *
 * @param preprocDefs Optional string with additional preprocessor definitions
 *
 * @param typeInfo Implementation specific type information from the caller that ensures that
 *                 shaders are cached uniquely with the type of layer they are used for
 *
 * @param context Link to context (which should be the current one) to compile shader for
 *
 * @return Shader pointer to compiled shader program (see warning)
 *
 * This function offers a convenient interface to compile a vertex/fragment shader pair with some
 * additional benefits:
 *   1. Provided resource names are loaded from the resource system
 *   2. Resulting shader sources are preprocessed
 *   2. Shader sources are compiled and cached
 *   3. Resulting shader program is cached
 *
 * The preprocessing includes adding the provided \p preprocDefs after the shader preamble and
 * also resolving any \c \#include statements in the shader sources by substituting the right
 * ShaderSnippet for that. If a shader cache is available, the preprocessed shader sources are
 * then checked for presence in the cache, in which case the already compiled shader is taken from
 * the cache. Otherwise the shaders are compiled and put into the shader cache. Following the
 * compilation / cache-lookup, it is checked if the shaders are already available as a linked
 * program for this type of layer (this is where the \p typeInfo parameter comes into play). If
 * that is not the case, the shaders are linked and put into the cache, otherwise the cached
 * instance is used.
 *
 * The main reason for the \p typeInfo parameter is to make sure that shader \e programs are not
 * cached between different types of layer as some static settings on the uniform variables may
 * differ. It is up to the implementation of the actual layers to make sure that uniforms which
 * are not exactly the same based on the layer type (e.g. they are dependent on image resolution),
 * are re-set before running the shader
 *
 * @warning This function does \b not link the resulting shader program and it is up to the
 *          caller to make sure of that. However, when calling this function with a set of shaders
 *          for which a shader program was already cached, the returned shader program <i>might
 *          already be linked</i>. Make sure to query the shader state before conducting operations
 *          that require a certain state.
 *
 * @throws ShaderException or GLException in case of errors
 *
 * @see ShaderRepository::getShader, ShaderCache::findShader
 */
programptr ShaderRepository::compileShaderPair(const char *vertexName, const char *fragmentName,
                                               const char *preprocDefs, const std::type_info& typeInfo,
                                               const fyusenet::GfxContextLink& context) {
    using namespace fyusion::opengl;
    const char *vert = ShaderRepository::getShader(vertexName);
    const char *frag = ShaderRepository::getShader(fragmentName);
    if (!vert) THROW_EXCEPTION_ARGS(ShaderException, "Cannot load vertex shader %s (not found)", vertexName);
    if (!frag) THROW_EXCEPTION_ARGS(ShaderException, "Cannot load fragment shader %s (not found)", fragmentName);
    shaderptr vshader(new VertexShader(context));
    shaderptr fshader(new FragmentShader(context));
    vshader->setResourceName(vertexName);
    fshader->setResourceName(fragmentName);
    vshader->setCode(vert);
    fshader->setCode(frag);
    vshader->setPreprocDefs(preprocDefs);
    fshader->setPreprocDefs(preprocDefs);
    ShaderCache *cache = ShaderCache::getInstance(context);
    if (cache) {
        size_t modhash = typeInfo.hash_code();
        shaderptr vcache = cache->findShader(vshader);
        shaderptr fcache = cache->findShader(fshader);
        if (vcache && fcache) {
            std::vector<GLuint> handles{vcache->getHandle(), fcache->getHandle()};
            programptr prog = cache->findProgram(modhash, handles);
            if (prog) {
                return prog;
            }
        }
        programptr prog = ShaderProgram::createInstance(context);
        prog->addShader( (vcache) ? vcache : vshader );
        prog->addShader( (fcache) ? fcache : fshader );
        prog->compile();
        if ((!vcache) && (cache)) cache->putShader(vshader);
        if ((!fcache) && (cache)) cache->putShader(fshader);
        if (cache) cache->putProgram(prog, modhash);
        return prog;
    } else {
        programptr prog = ShaderProgram::createInstance(context);
        prog->addShader(vshader);
        prog->addShader(fshader);
        prog->compile();
        return prog;
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Idle constructor
 */
ShaderRepository::ShaderRepository() {
}


/**
 * @brief Retrieve reference to ShaderRepository singleton
 *
 * @return ShaderRepository singleton that contains all registered shader sources
 */
ShaderRepository & ShaderRepository::repository() {
    static ShaderRepository repos;
    return repos;
}


/**
 * @brief Add shader source to resource system
 *
 * @param shader Source code of the shader to add
 *
 * @param resourceName Name of the shader in the resource system
 */
void ShaderRepository::addResource(const char *shader,const char *resourceName) {
    std::string key(resourceName);
    shaderMap_[key]=shader;
}

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
