//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GL Texture Loader for Linear Data (Header)                                  (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------

namespace fyusion::fyusenet::gpu::rudiments {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Helper class for loading weight/bias textures into linear layers
 */
class LinearTextureLoader {
 public:
    static void loadRM4BitQuantizedWeights(const uint32_t * weights, int rows, int columns, GLuint wgtTex);

    template<typename T, GLint gpulayout, GLenum cpulayout, GLenum cputype>
    static void load4BitQuantizationTables(const T * scales, const uint32_t *qZeros,
                                           int rows, int columns, int quantGroupSize,
                                           GLuint scaleTex, GLuint zeroTex);

 private:
    static void bindTexture(GLuint texture);
};

} // fyusion:.fyusenet::gpu::rudiments namespace

// vim: set expandtab ts=4 sw=4:

