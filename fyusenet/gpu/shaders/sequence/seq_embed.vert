/* -------------------------------------------------------------------------------------------------
 * Embedding layer for Sequences                                               (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#else
precision highp float;
precision highp int;
precision highp sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform highp usampler2D inputTokens;
#else
uniform highp usampler2D inputTokens;
#endif

in highp uint attributes0;

out highp float textureX;
flat out highp ivec2 textureYZ;

uniform highp ivec2 viewport;
uniform highp int textureHeight;

void main(void) {
    int row = int(attributes0 >> 1);
    int col = int(attributes0 & 0x1u);
    float fuzz = (float(col) * 0.5) / float(viewport.x);
    highp uint tidx = texelFetch(inputTokens, ivec2(0, row), 0).r;
    gl_Position = vec4((float(2*col)+fuzz)-1.0,2.0*((float(row)+0.25)/float(viewport.y))-1.0, 0.0, 1.0);  // see "diamond-exit" rule in GL spec section 3.4.1
    textureYZ = ivec2(tidx % uint(textureHeight),  tidx / uint(textureHeight));
    textureX = float(col) * float(viewport.x);
}

