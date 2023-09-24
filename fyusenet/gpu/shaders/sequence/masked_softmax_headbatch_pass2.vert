/* -------------------------------------------------------------------------------------------------
 * Masked SoftMax for Self-Attention Layer (Pass 2/2)                          (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;

in highp vec3 attributes0;          // x,y: tex coordinate ; z: 0 for top of tile, 1 for bottom of tile

out highp vec3 tileCoord;           // x,y: dot-prod texture input coordinate ; z: 1st mask-out position

uniform highp vec4 viewport;        // x,y: viewport size ; z,w: viewport scaling
uniform highp ivec2 inputParams;    // x: key length,  y: query length
uniform highp int baseTokenIdx;     // index of the first token (corresponding to generic y == 0)

void main(void) {
    vec2 npos = attributes0.xy * viewport.zw;
    gl_Position = vec4(npos * 2.0 - vec2(1.0), 0.0, 1.0);
    tileCoord.xy = npos * viewport.xy;
    tileCoord.z = float(baseTokenIdx) + attributes0.z * float(inputParams.y);
}

