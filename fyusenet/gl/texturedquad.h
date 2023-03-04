//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Simple textured quadrilateral (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "vao.h"
#include "vbo.h"
#include "../gpu/gfxcontextlink.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Class for instantiating and drawing a simple textured quad (viewport filling)
 *
 * This class creates a quadrilateral that can be attached to an existing %VAO
 */
class TexturedQuad : public fyusenet::GfxContextTracker {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    TexturedQuad(const fyusenet::GfxContextLink & ctx = fyusenet::GfxContextLink(), bool flipY=false);
    virtual ~TexturedQuad();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void init(VAO *vao, int arrayIdx = 0);
    void cleanup();

    /**
     * @brief Draw quad to screen
     *
     * @pre The %VAO that was used in the initialization is currently active
     */
    void draw() {
        glDrawArrays(GL_TRIANGLE_FAN,0,4);
    }

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupQuad(VAO *vao, int arrayIdx);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VBO * quad_ = nullptr;              //!< Buffer objcet that holds the quad coordinates
    bool vertFlip_ = false;             //!< If true, will invert the quad vertically
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
