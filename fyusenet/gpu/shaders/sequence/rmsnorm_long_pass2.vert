/* -------------------------------------------------------------------------------------------------
 * RMSNormLayer for Sequence Layouts for m tokens (Pass 2/2)                        (c) Martin Wawro 2023
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

in highp vec2 attributes0;

out highp vec2 texCoord;

uniform highp vec2 viewport;

void main(void) {
    highp vec2 nrmatt = (attributes0.xy + vec2(1.0)) * 0.5;
    gl_Position = vec4(attributes0.xy, 0.0, 1.0);
    texCoord = viewport * nrmatt;
}

