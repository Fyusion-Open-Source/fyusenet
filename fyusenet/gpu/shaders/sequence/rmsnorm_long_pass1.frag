/* -------------------------------------------------------------------------------------------------
 * RMSNormLayer for Sequence Layouts (Pass 1/2)                                     (c) Martin Wawro 2023
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
#else
uniform sampler2D inputLayer0;
#endif

layout(location=0) out float fragmentColor0;

in highp vec2 basePos;
uniform highp int contraction;

void main(void) {
    vec4 accu = vec4(0);
    ivec2 xy = ivec2(basePos);
    for (int i=0; i < contraction; i++) {
        vec4 val = texelFetch(inputLayer0, xy + ivec2(i,0), 0);
        accu += val * val;
    }
    fragmentColor0 = dot(accu, vec4(1.0));
}
