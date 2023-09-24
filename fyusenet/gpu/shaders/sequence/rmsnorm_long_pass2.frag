/* -------------------------------------------------------------------------------------------------
 * RMSNormLayer for Sequence Layouts for m tokens (Pass 2/2)                        (c) Martin Wawro 2023
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

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
layout(binding=1) uniform highp sampler2D normData;
layout(binding=2) uniform highp sampler2D weights;
#else
uniform sampler2D inputLayer0;
uniform highp sampler2D normData;
uniform sampler2D weights;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 texCoord;
uniform highp float scale;

// NOTE (mw) wasteful 2nd fetch and computation here, maybe optimize later by lines
// and pass the normalization values from the vertex shader
void main(void) {
    ivec2 pos = ivec2(texCoord.xy);
    vec4 num = texelFetch(inputLayer0, pos, 0);
    vec4 wgt = texelFetch(weights, ivec2(pos.x, 0), 0);
    float denom = inversesqrt(1.0e-6 + texelFetch(normData, ivec2(pos.y, 0), 0).r * scale);
    fragmentColor0 = wgt * num * denom;
}
