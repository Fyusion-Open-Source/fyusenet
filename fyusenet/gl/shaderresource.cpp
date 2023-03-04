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
