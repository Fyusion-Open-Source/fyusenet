//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Snippet for Custom Include Statements
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "shadersnippet.h"
#include "shaderresource.h"

//-------------------------------------- Global Variables ------------------------------------------
namespace fyusion {
namespace opengl {

//-------------------------------------- Local Definitions -----------------------------------------


std::unordered_map<std::string,ShaderSnippet *> ShaderSnippet::repository_;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param code Source code string to wrap snippet around
 */
ShaderSnippet::ShaderSnippet(const std::string& code):code_(code) {
}


/**
 * @brief Load shader snippet from resource system by name
 *
 * @param resName Name of the shader snippet in the resource system
 *
 * @return Pointer to ShaderSnippet object
 *
 * Loads a shader snippet based on its resource name.
 *
 * @see ShaderRepository::getShader
 */
const ShaderSnippet * ShaderSnippet::loadFromResource(const std::string & resName) {
    auto ri = repository_.find(resName);
    if (ri != repository_.end()) return ri->second;
    else {
      const char * code = ShaderRepository::getShader(resName.c_str());
      if (!code) return nullptr;
      repository_[resName] = new ShaderSnippet(std::string(code));
      return repository_[resName];
    }
}

/**
 * @brief Release memory resources in shader snippet storage
 */
void ShaderSnippet::tearDown() {
    for (auto ri = repository_.begin(); ri != repository_.end(); ++ri) {
        delete ri->second;
    }
    repository_.clear();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
