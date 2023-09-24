/* -------------------------------------------------------------------------------------------------
 * Attention-Layer QK Dot-Product (batched)                                    (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;

in highp vec4 attributes0;       // x,y: ndc coordinates, z: head offset (div 4), w: 0 on top edge, 1 on bottom edge

out highp vec2 inputPos;
flat out highp int headIdx;
flat out int innerBatch;

uniform highp vec4 viewport;     // x,y: viewport size, z,w: viewport scaling
uniform highp ivec4 sizeParams;  // x: head-size (pixels), y: #heads, z: #key tokens, w: #query tokens
uniform highp int headOffset;    //

void main(void) {
    gl_Position = vec4((((attributes0.xy + vec2(1.0)) / 2.0) * viewport.zw * 2.0) - vec2(1.0), 0.0, 1.0);
    vec2 ipos = (attributes0.xw + vec2(1.0, 0.0)) / vec2(2.0, 1.0);
    inputPos.xy = ipos * vec2(sizeParams.zw);
    headIdx = headOffset + int(attributes0.z);
    innerBatch = gl_InstanceID;
}

