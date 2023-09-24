/* -------------------------------------------------------------------------------------------------
 * Masked SoftMax for Self-Attention Layer (Pass 2/2)                          (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;

in highp vec2 attributes0;          // x,y: ndc coordinate

out highp vec2 tileCoord;           // x,y: dot-prod texture input coordinate

uniform highp vec2 viewport;        // x,y: viewport size ; z,w: viewport scaling
uniform highp ivec2 inputParams;    // x: key length,  y: query length

void main(void) {
    gl_Position = vec4(attributes0.xy, 0.0, 1.0);
    vec2 npos = (attributes0.xy + vec2(1.0)) * 0.5;
    tileCoord.xy = npos * viewport.xy;
}

