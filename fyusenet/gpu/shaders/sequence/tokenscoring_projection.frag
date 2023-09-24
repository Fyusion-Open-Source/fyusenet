/* -------------------------------------------------------------------------------------------------
 * Projection shader for token scoring                                         (c) Martin Wawro 2023
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

#ifdef BINDING_SUPPORT
layout(binding=1) uniform sampler2D vocabulary;
#else
uniform sampler2D vocabulary;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 tokenXY;          // x: base x-position of input token, y: y-position of input token
in highp vec2 outPosition;		// x: x-position within vp, y: y-position within vp

flat in vec4 instanceData[INSTANCE_WIDTH];

uniform highp ivec2 viewport;

void main(void) {
    ivec2 opos = ivec2(outPosition);
    int vocabrow = (opos.x + opos.y * viewport.x) * 4;
    vec4 accu = vec4(0);
    for (int i=0; i < INSTANCE_WIDTH; i++) {
        vec4 inem = instanceData[i];
        int x = int(tokenXY.x) + i;
        vec4 tab0 = texelFetch(vocabulary, ivec2(x, vocabrow), 0);
        vec4 tab1 = texelFetch(vocabulary, ivec2(x, vocabrow+1), 0);
        vec4 tab2 = texelFetch(vocabulary, ivec2(x, vocabrow+2), 0);
        vec4 tab3 = texelFetch(vocabulary, ivec2(x, vocabrow+3), 0);
        accu += vec4(dot(inem, tab0), dot(inem, tab1), dot(inem, tab2), dot(inem, tab3));
    }
    fragmentColor0 = accu;
}

