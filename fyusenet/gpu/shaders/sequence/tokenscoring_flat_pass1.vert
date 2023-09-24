/* -------------------------------------------------------------------------------------------------
 * Pass 1 Flattening for Token Scoring                                         (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#else
precision highp float;
precision highp int;
precision highp sampler2D;
#endif

in highp vec4 attributes0;

out highp vec2 texturePos;

uniform highp ivec2 textSize;
uniform highp vec2 shift;

void main(void) {
    gl_Position = vec4(attributes0.xy + shift, 0.0, 1.0);
    texturePos = attributes0.zw * vec2(textSize);
    gl_PointSize = 1.0;
}

