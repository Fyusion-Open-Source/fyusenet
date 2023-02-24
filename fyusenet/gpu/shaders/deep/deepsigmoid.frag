/* ----------------------------------------------------------------------------
 * Sigmoid Shader (Deep)                   Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

#include "shaders/activation.inc"

void main(void) {
    vec4 data = activate(texture(inputLayer0,texCoord.xy));
    fragmentColor0 = 1.0/(1.0+exp(-data));
}
