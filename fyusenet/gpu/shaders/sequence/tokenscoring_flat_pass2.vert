/* -------------------------------------------------------------------------------------------------
 * Token Selection/Sampling Flattening Pass 2/2                                (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump sampler2D;
#else
precision highp sampler2D;
#endif
precision highp float;
precision highp int;

in highp uint attributes0;

flat out highp vec4 pixData;

void main(void) {
    gl_Position = vec4(float(gl_VertexID)-0.5, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}

