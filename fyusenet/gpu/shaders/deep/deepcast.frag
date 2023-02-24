/* ----------------------------------------------------------------------------
 * Type-Cast Shader (Deep)                 Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

#include "shaders/activation.inc"
#include "shaders/typecast.inc"

void main(void) {
    vec4 data = activate(texture(inputLayer0,texCoord.xy));
#if !defined(CAST_TO_FLOAT16) && !defined(CAST_TO_FLOAT32)
    // NOTE (mw) we are not resorting to integral textures, instead we do truncation/rounding
    // and stay in the floating-point world by casting back to an FP type
    fragmentColor0 = vec4(castTo(data));
#else
    return vec4(data);
#endif
}
