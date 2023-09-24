/* -------------------------------------------------------------------------------------------------
 * Token-Scoring Selection                                                     (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

in highp uint attributes0;

void main(void) {
    // selection shader is a single point (for now)
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}

