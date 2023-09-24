/* -------------------------------------------------------------------------------------------------
 * Token-Scoring Scatter Operation Pass 1                                      (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

layout(location=0) out uint fragmentColor0;
layout(location=1) out vec4 fragmentColor1;

flat in highp uint tokenIndex;
flat in highp vec4 match;

void main(void) {
    fragmentColor0 = tokenIndex;
    fragmentColor1 = match;
}
