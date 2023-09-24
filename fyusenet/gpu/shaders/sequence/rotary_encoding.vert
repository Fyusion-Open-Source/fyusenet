/* -------------------------------------------------------------------------------------------------
 * Rotary Encoding as Positional Encoding for Sequences                        (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

in highp vec2 attributes0;

out highp vec2 inputPos;

uniform ivec2 viewport;       // x,y: viewport size (pixels)

void main(void) {
    gl_Position = vec4(attributes0.xy, 0.0, 1.0);
    inputPos = (((attributes0.xy + vec2(1.0)) / 2.0) * vec2(viewport.xy));
}

