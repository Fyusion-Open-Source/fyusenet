/* -------------------------------------------------------------------------------------------------
 * Matrix-Multiplication for Sequences (short)                                 (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#ifndef INSTANCE_OFFSET
#define INSTANCE_OFFSET 0
#endif

precision highp float;
precision highp int;
precision highp sampler2D;

in highp uint attributes0;

out highp vec2 inputPos;
flat out highp ivec2 colOffset;

uniform highp ivec2 viewport;       // x,y: viewport size

void main(void) {
    int row = int(attributes0 >> 16);
    int column = int(attributes0 & 0xFFFFu) * viewport.x;
    float xfuzz = (column > 0) ? 0.25 : 0.0;
    // -----------------------------------------------------
    // Determine the points in the output (either line
    // endpoints or single points)...
    // -----------------------------------------------------
    vec2 cr = (vec2(column,row) + vec2(xfuzz, 0.25)) / vec2(viewport.xy);       // see "diamond-exit" rule in GL spec section 3.4.1
    gl_Position = vec4((cr * 2.0) - vec2(1.0), 0.0, 1.0);
    // TODO (mw) prefetch data at coloffset to get rid of texture fetches in fragment shader
    colOffset = ivec2((gl_InstanceID + INSTANCE_OFFSET) * MATRIX_WEIGHTS * MATRIX_PACKS,
                      (gl_InstanceID + INSTANCE_OFFSET) * MATRIX_PACKS);
    inputPos.xy = vec2(column, row);
}

