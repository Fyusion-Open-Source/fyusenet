//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Snippet for Custom Include Statements (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Representation of shader snippet code
 *
 * This class wraps shader code that can be included (non-recursively) by shaders using an internal
 * \c \#include statement which does not exist in GLSL shaders. Shader snippets use a \c .inc
 * extension and are also part of the shader resource system. The following example shows
 * how a snippet can be used:
 * @code
 *  ...
 *  uniform sampler2D mytex;
 *
 *  #include "shaders/mysnippet.inc"
 *
 *  void main() {
 *      ...
 *  }
 * @endcode
 *
 * The include statement will be replaced by the content of the shader snippet in the source
 * before passing the source to the GLSL compiler. This allows for better re-use of recurrent
 * parts in shaders.
 */
class ShaderSnippet {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ShaderSnippet(const std::string& code);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    static const ShaderSnippet * loadFromResource(const std::string& resName);
    static void tearDown();

    /**
     * @brief Get source code of snippet
     *
     * @return String with source code of the shader snippet wrapped by this object
     */
    const std::string& code() const {
      return code_;
    }
 private:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    static std::unordered_map<std::string, ShaderSnippet *> repository_;   //!< Shader snippet repository (singleton)
    std::string code_;                                                     //!< Source code of snippet
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
