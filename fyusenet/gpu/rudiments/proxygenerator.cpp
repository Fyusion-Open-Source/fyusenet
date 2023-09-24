//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Generic Proxy Geometry Builder                                              (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <algorithm>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "proxygenerator.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::rudiments {
//-------------------------------------- Local Definitions -----------------------------------------

const GLfloat ProxyGenerator::texQuadVerts_[] = {-1.0f, -1.0f, 0.0f, 0.0f,
                                                  1.0f, -1.0f, 1.0f, 0.0f,
                                                  1.0f,  1.0f, 1.0f, 1.0f,
                                                 -1.0f,  1.0f, 0.0f, 1.0f};

const GLfloat ProxyGenerator::quadVerts_[] = {-1.0f, -1.0f,
                                               1.0f, -1.0f,
                                               1.0f,  1.0f,
                                              -1.0f,  1.0f};

const GLshort ProxyGenerator::quadIndices_[] = {0, 1, 2, 0, 2, 3};

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Generate a viewport-filling quadrilateral with texture coordinates packed in 4-vec format
 *
 * @param context Graphics context link to use
 *
 * @return Tuple of VAO, VBO and IBO that refer to the generated geometry
 *
 * This generates a simple quad in the form of two triangles which is screen filling [-1, 1] in NDC
 * and applies [0,1] texture coordinates to those. Each vertex is stored as 4-vec with the following
 * layout:
 *   - x: x coordinate in NDC
 *   - y: y coordinate in NDC
 *   - z: s coordinate in normalized texture coordinates
 *   - w: s coordinate in normalized texture coordinates
 */
std::tuple<opengl::VAO *, opengl::VBO *, opengl::IBO *> ProxyGenerator::texturedQuad(const GfxContextLink& context) {
    using namespace opengl;
    auto * array = new VAO(context);
    array->bind();
    auto * vertices = new VBO(context);
    array->enableArray(0);
    vertices->setBufferData((void *)texQuadVerts_, 4 * 4 * sizeof(GLfloat), GL_STATIC_DRAW);
    vertices->bind();
    array->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    auto * indices = new IBO(context);
    indices->setBufferData((void *)quadIndices_, 6 * sizeof(GLshort), GL_STATIC_DRAW);
    indices->bind();
    array->unbind();
    return {array, vertices, indices};
}


/**
 * @brief Generate a viewport-filling quadrilateral packed in vec2-format
 *
 * @param context Graphics context link to use
 *
 * @return Tuple of VAO, VBO and IBO that refer to the generated geometry
 *
 * This generates a simple quad in the form of two triangles which is screen filling [-1, 1] in NDC.
 * Each vertex is stored as 2-vec with the following layout:
 *   - x: x coordinate in NDC
 *   - y: y coordinate in NDC
 */
std::tuple<opengl::VAO *, opengl::VBO *, opengl::IBO *> ProxyGenerator::simpleQuad(const GfxContextLink& context) {
    using namespace opengl;
    auto * array = new VAO(context);
    array->bind();
    auto * vertices = new VBO(context);
    array->enableArray(0);
    vertices->setBufferData((void *)quadVerts_, 2 * 4 * sizeof(GLfloat), GL_STATIC_DRAW);
    vertices->bind();
    array->setVertexAttributeBuffer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    auto * indices = new IBO(context);
    indices->setBufferData((void *)quadIndices_, 6 * sizeof(GLshort), GL_STATIC_DRAW);
    indices->bind();
    array->unbind();
    return {array, vertices, indices};
}


/**
 * @brief Create a set of vertically stacked textured quads
 *
 * @param context Graphics context link to use
 * @param heights Vector of individual heights of the generated strips (in pixels)
 * @param fullHeight Full height of target viewport (in pixels)
 *
 * @return Tuple of VAO, VBO and IBO that refer to the generated geometry
 *
 * This function generates a set of quads (rendered by two triangles each) that are stacked
 * vertically. The full height as well as the stack heights are given in pixels and are
 * normalized to NDCs and the texture coordinates to the [0,1] interval. The vertices are laid out
 * as 4-vecs laid out as follows:
 *   - x: x coordinate in NDC
 *   - y: y coordinate in NDC
 *   - z: s coordinate in normalized texture coordinates
 *   - w: s coordinate in normalized texture coordinates
 */
std::tuple<opengl::VAO *, opengl::VBO *, opengl::IBO *> ProxyGenerator::verticalTexturedQuads(const GfxContextLink& context, const std::vector<int>& heights, int fullHeight) {
    using namespace opengl;
    auto * array = new VAO(context);
    array->bind();
    auto * vertices = new VBO(context);
    std::unique_ptr<float[]> vdata(new float[heights.size() * 4 * 4]);
    float * ptr = vdata.get();
    for (size_t tile=0,yoffset=0; tile < heights.size(); tile++) {
        float y0 = (float)(yoffset) / (float)fullHeight;
        float y1 = (float)(yoffset + heights[tile]) / (float)fullHeight;
        *ptr++ = -1.0f; *ptr++ = 2.0f * y0 - 1.0f; *ptr++ = 0.0f; *ptr++ = 0.0f;
        *ptr++ =  1.0f; *ptr++ = 2.0f * y0 - 1.0f; *ptr++ = 1.0f; *ptr++ = 0.0f;
        *ptr++ =  1.0f; *ptr++ = 2.0f * y1 - 1.0f; *ptr++ = 1.0f; *ptr++ = 1.0f;
        *ptr++ = -1.0f; *ptr++ = 2.0f * y1 - 1.0f; *ptr++ = 0.0f; *ptr++ = 1.0f;
    }
    array->enableArray(0);
    vertices->setBufferData((void *)vdata.get(), (int)(heights.size() * 4 * sizeof(GLfloat)), GL_STATIC_DRAW);
    vertices->bind();
    array->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    std::unique_ptr<GLshort[]> indices(new GLshort[heights.size() * 6]);
    GLshort * iptr = indices.get();
    for (size_t tile=0; tile < heights.size(); tile++) {
        *iptr++ = (GLshort)(tile * 4 + 0);
        *iptr++ = (GLshort)(tile * 4 + 1);
        *iptr++ = (GLshort)(tile * 4 + 2);
        *iptr++ = (GLshort)(tile * 4 + 0);
        *iptr++ = (GLshort)(tile * 4 + 2);
        *iptr++ = (GLshort)(tile * 4 + 3);
    }
    auto * indbo = new IBO(context);
    indbo->setBufferData((void *)quadIndices_, 6 * sizeof(GLshort), GL_STATIC_DRAW);
    indbo->bind();
    array->unbind();
    return {array, vertices, indbo};
}


/**
 * @brief Create a set of point primitives laid out in a grid
 *
 * @param context Graphics context link to use
 * @param columns Number of columns for the dot grid
 * @param rows Number of rows for the dot grid
 *
 * @return Tuple of VAO, VBO that refer to the generated geometry
 *
 * This function generates a grid of point primitives which are evenly spread within the full
 * NDC range ([-1,1]). Each point is stored as a 4-vec with the following layout:
 *   - x: x coordinate in NDC
 *   - y: y coordinate in NDC
 *   - z: s coordinate in normalized texture coordinates
 *   - w: s coordinate in normalized texture coordinates
 */
std::tuple<opengl::VAO *, opengl::VBO *> ProxyGenerator::texturedDotMatrix(const GfxContextLink& context, int columns, int rows) {
    using namespace opengl;
    auto * array = new VAO(context);
    array->bind();
    auto * vertices = new VBO(context);
    std::unique_ptr<float[]> vdata(new float[rows * columns * 4]);
    float * ptr = vdata.get();
    for (int y=0; y < rows; y++) {
        for (int x=0; x < columns; x++) {
            *ptr++ = ((columns > 1) ? -1.0f : 0.0f) + 2.0f * (float)x / (float)columns;
            *ptr++ = ((rows > 1) ? -1.0f : 0.0f) + 2.0f * (float)y / (float)rows;
            *ptr++ = (float)x / (float)columns;
            *ptr++ = (float)y / (float)rows;
        }
    }
    array->enableArray(0);
    vertices->setBufferData((void *)vdata.get(), (int)(rows * columns * 4 * sizeof(GLfloat)), GL_STATIC_DRAW);
    vertices->bind();
    array->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    array->unbind();
    return {array, vertices};
}


/**
 * @brief Create a set of point primitives laid out in a grid
 *
 * @param context Graphics context link to use
 * @param columns Number of columns for the dot grid
 * @param rows Number of rows for the dot grid
 *
 * @return Tuple of VAO, VBO that refer to the generated geometry
 *
 * This function generates a grid of point primitives which are evenly spread within the full
 * NDC range ([-1,1]). Each point is stored as a 2-vec with the following layout:
 *   - x: x coordinate in NDC
 *   - y: y coordinate in NDC
 */
std::tuple<opengl::VAO *, opengl::VBO *> ProxyGenerator::dotMatrix(const GfxContextLink& context, int columns, int rows) {
    using namespace opengl;
    auto * array = new VAO(context);
    array->bind();
    auto * vertices = new VBO(context);
    std::unique_ptr<float[]> vdata(new float[rows * columns * 2]);
    float * ptr = vdata.get();
    for (int y=0; y < rows; y++) {
        for (int x=0; x < columns; x++) {
            *ptr++ = ((columns > 1) ? -1.0f : 0.0f) + 2.0f * (float)x / (float)columns;
            *ptr++ = ((rows > 1) ? -1.0f : 0.0f) + 2.0f * (float)y / (float)rows;
        }
    }
    array->enableArray(0);
    vertices->setBufferData((void *)vdata.get(), (int)(rows * columns * 2 * sizeof(GLfloat)), GL_STATIC_DRAW);
    vertices->bind();
    array->setVertexAttributeBuffer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    array->unbind();
    return {array, vertices};
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // fyusion::fyusenet::gpu::rudiments namespace

// vim: set expandtab ts=4 sw=4:
