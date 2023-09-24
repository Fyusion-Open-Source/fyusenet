/* -------------------------------------------------------------------------------------------------
 * Attention-Layer QK Dot-Product (single query token)                         (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;

in highp vec4 attributes0;       // x,y: ndc coordinates, z,w: tex coordinates

out highp vec2 keyHeadPos;
flat out int innerBatch;

uniform highp vec2 viewport;      // x,y: viewport size, z,w: viewport scaling
uniform highp ivec4 inputParams;  // x: head-size (pixels), y: #heads (pixels), z: #key tokens, w: #query tokens

void main(void) {
    gl_Position = vec4(attributes0.xy, 0.0, 1.0);
    vec2 ipos = attributes0.zw;
    keyHeadPos.x = attributes0.z * float(inputParams.z);
    keyHeadPos.y = attributes0.w * float(inputParams.y);
    innerBatch = gl_InstanceID;
}

