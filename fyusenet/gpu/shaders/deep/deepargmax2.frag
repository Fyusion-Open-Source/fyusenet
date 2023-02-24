/* ----------------------------------------------------------------------------
 * ArgMax Shader (Deep)                    Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

uniform highp int bitmask;

void main(void) {
    highp vec2 pix = texture(inputLayer0, texCoord).rg;
    highp int ipix = floatBitsToInt(pix.y);
    highp int idx = (ipix & bitmask) >> PLACEMENT_BITS;
    fragmentColor0.rg = vec2(idx, pix.g);
}
