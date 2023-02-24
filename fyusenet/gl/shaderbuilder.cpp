//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Shader-(Pair) Builder
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "shaderbuilder.h"
#include "vertexshader.h"
#include "fragmentshader.h"
#include "shadercache.h"
#include "shaderexception.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Compile and link vertex/fragment shader pair from the resource system (cache-aware)
 *
 * @param vertResName Resource name of the vertex shader
 * @param fragResName Resource name of the fragment shader
 * @param typeInfo Type information of the class that requests the compile/link (used for caching)
 * @param extraDefs Additional pre-processor definitions to be placed in the shader
 * @param context Link to GL context for the shaders to be compiled under
 *
 * @return Shared pointer to compiled and linked shader program, ready-to-use.
 *
 * @throws ShaderException in case the shader compilation/linkage fails, or the shaders were not
 *         found within the resource system
 *
 * This function goes through all the steps to load the two specified shader sources from the
 * resource system, checks if the shader(s) were already cached and uses a cached instance
 * in that case. Otherwise the shaders are compiled, placed into the cache and subsequently
 * linked (and the resulting program is also put into the cache).
 *
 * Note that the \p typeInfo parameter is used for the hash computation (which also includes the
 * \p extraDefs parameter as well as the source code of the shader(s) itself).
 *
 * @see ShaderCache, ShaderRepository
 */
programptr ShaderBuilder::shaderProgramFromResource(const char *vertResName, const char *fragResName, const std::type_info& typeInfo, const char *extraDefs, const fyusenet::GfxContextLink& context) {
    const char * vert = ShaderRepository::getShader(vertResName);
    const char * frag = ShaderRepository::getShader(fragResName);
    if (!vert) THROW_EXCEPTION_ARGS(ShaderException,"Cannot load vertex shader %s (not found)",vertResName);
    if (!frag) THROW_EXCEPTION_ARGS(ShaderException,"Cannot load fragmnet shader %s (not found)",fragResName);
    shaderptr vshader(new VertexShader(context));
    shaderptr fshader(new FragmentShader(context));
    vshader->setResourceName(vertResName);
    fshader->setResourceName(fragResName);
    vshader->setCode(vert);
    fshader->setCode(frag);
    vshader->setPreprocDefs(extraDefs);
    fshader->setPreprocDefs(extraDefs);
    ShaderCache *cache = ShaderCache::getInstance(context);
    try {
        if (cache) {
            size_t modhash = typeInfo.hash_code();
            shaderptr vcache = cache->findShader(vshader);
            shaderptr fcache = cache->findShader(fshader);
            if (vcache && fcache) {
                std::vector<GLuint> handles{vcache->getHandle(),fcache->getHandle()};
                programptr prog = cache->findProgram(modhash,handles);
                if (prog) {
                    return prog;
                }
            }
            programptr prog = ShaderProgram::createInstance();
            prog->addShader( (vcache) ? vcache : vshader );
            prog->addShader( (fcache) ? fcache : fshader );
            prog->compile();
            prog->link();
            if ((!vcache)&&(cache)) cache->putShader(vshader);
            if ((!fcache)&&(cache)) cache->putShader(fshader);
            if (cache) cache->putProgram(prog,modhash);
            return prog;
        } else {
            programptr prog = ShaderProgram::createInstance();
            prog->addShader(vshader);
            prog->addShader(fshader);
            prog->compile();
            prog->link();
            return prog;
        }
    } catch (GLException& ex) {
        FNLOGE("Cannot compile shader pair %s / %s", vertResName, fragResName);
        throw;
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
