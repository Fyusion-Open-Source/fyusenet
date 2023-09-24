/* -------------------------------------------------------------------------------------------------
 * Masked SoftMax for Self-Attention Layer (Pass 2/2)                          (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
layout(binding=1) uniform sampler2D inputLayer1;
#else
uniform sampler2D inputLayer0;
uniform sampler2D inputLayer1;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 tileCoord;      // x,y input tile coordinate
uniform highp int tokenIdx;   // index of the query token token

void main(void) {
    ivec2 tilepos = ivec2(tileCoord.xy);
    if (tilepos.x > tokenIdx) discard;
    vec4 data = texelFetch(inputLayer0, tilepos, 0);
    vec4 denom = texelFetch(inputLayer1, ivec2(0, tilepos.y), 0);
    fragmentColor0 = exp(data) / denom;
}
