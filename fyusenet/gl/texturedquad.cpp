//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Simple textured quadrilateral
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "texturedquad.h"

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
 * @param ctx Link to GL context to be used for this quad
 *
 * @param flipY If \c true, will invert the quad vertically
 */
TexturedQuad::TexturedQuad(const fyusenet::GfxContextLink & ctx, bool flipY) : vertFlip_(flipY) {
    setContext(ctx);
}


/**
 * @brief Idle destructor
 */
TexturedQuad::~TexturedQuad() {
    assert(quad_ == nullptr);
}

/**
 * @brief Release GL resources held by this instance
 */
void TexturedQuad::cleanup() {
    if (quad_) delete quad_;
    quad_ = nullptr;
}

/**
 * @brief Initialize GL resources for drawing a quadrilateral
 *
 * @param vao Pointer to VAO object that compounds the buffer objects used for the quad
 *
 * @param arrayIdx Index of the to-be-created quad VBO within the supplied VAO
 */
void TexturedQuad::init(VAO *vao,int arrayIdx) {
    setupQuad(vao,arrayIdx);
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Create VBO for drawing quad
 *
 * @param vao Vertex array object to add the created VBO to
 *
 * @param arrayIdx Index of the VBO within the supplied VAO
 *
 * This creates a simple 2D quadrilateral that covers the full viewport
 */
void TexturedQuad::setupQuad(VAO *vao, int arrayIdx) {    
    const int vertsize = 4;
    float tmp[vertsize*4];
    float posleft = -1.0f;
    float posright = 1.0f;
    float postop = (vertFlip_) ? 1.0f : -1.0f;
    float posbottom = (vertFlip_) ? -1.0f : 1.0f;
    float tleft = 0.0f;
    float ttop = 0.0f;
    float thspan = 1.0f;
    float tvspan = 1.0f;
    tmp[0*vertsize+0] = posleft;       // position (topleft)
    tmp[0*vertsize+1] = postop;
    tmp[0*vertsize+2] = tleft;         // texture
    tmp[0*vertsize+3] = ttop;
    tmp[1*vertsize+0] = posleft;       // position (bottomleft)
    tmp[1*vertsize+1] = posbottom;
    tmp[1*vertsize+2] = tleft;         // texture
    tmp[1*vertsize+3] = ttop+tvspan;
    tmp[2*vertsize+0] = posright;      // position (bottomright)
    tmp[2*vertsize+1] = posbottom;
    tmp[2*vertsize+2] = tleft+thspan;  // texture
    tmp[2*vertsize+3] = ttop+tvspan;
    tmp[3*vertsize+0] = posright;      // position (topright)
    tmp[3*vertsize+1] = postop;
    tmp[3*vertsize+2] = tleft+thspan;  // texture
    tmp[3*vertsize+3] = ttop;
    delete quad_;
    quad_ = new VBO(context_);
    vao->bind();
    vao->enableArray(arrayIdx);
    quad_->setBufferData(tmp,vertsize*4*sizeof(float) ,GL_STATIC_DRAW);
    quad_->bind();
    vao->setVertexAttributeBuffer(arrayIdx, vertsize, GL_FLOAT, GL_FALSE, 0, 0);
    vao->unbind();
}

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
