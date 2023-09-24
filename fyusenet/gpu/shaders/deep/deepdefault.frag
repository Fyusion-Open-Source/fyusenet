/* ----------------------------------------------------------------------------
 * Generic Activated Passthrough (deep)     Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

#include "shaders/activation.inc"

void main(void) {
    fragmentColor0 = activate(texture(inputLayer0,texCoord.xy));
    gl_PointSize = 1.0;
}
