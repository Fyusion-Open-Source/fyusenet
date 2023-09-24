/* -------------------------------------------------------------------------------------------------
 * RMSNormLayer for Sequence Layouts for m tokens (Pass 1/2)                        (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
#ifndef HIGH_PRECISION
precision mediump sampler2D;
#else
precision highp sampler2D;
#endif


in highp vec2 attributes0;

out highp vec2 basePos;

uniform highp int contraction;
uniform highp vec2 inputSize;

void main(void) {
    gl_Position = vec4((attributes0.yx + vec2(0, 0.5)) * 2.0 - vec2(1.0), 0.0, 1.0);   // flip line to horizontal
    float noffset = float(contraction * gl_InstanceID);
    basePos = vec2(noffset, attributes0.y * inputSize.y);
}

