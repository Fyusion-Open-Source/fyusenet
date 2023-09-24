/* -------------------------------------------------------------------------------------------------
 * Attention-Layer Attention Weights / Value Multiplication (Single Query)     (c) Martin Wawro 2023
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

in highp vec2 attributes0;

#ifdef BINDING_SUPPORT
layout(binding=1) uniform sampler2D attWeights;
#else
uniform sampler2D attWeights;
#endif

flat out vec4 weights[MATRIX_WEIGHTS];  //
flat out highp ivec2 weightData;        // x: #weights to use, y: row-offset in value data
out highp float valPos;                 // column in value data

uniform highp ivec2 viewport;           // x,y: viewport size
uniform highp int tokenIdx;             // index of query token in the query context

void fetchWeights(in ivec2 pos, in int nweights, in int head) {
    int numpacks = (nweights + 3) / 4;
    for (int i=0; i < numpacks; i++) {
        vec4 wgt = vec4(texelFetch(attWeights, pos + ivec2(i*4  , 0), 0)[head],
                        texelFetch(attWeights, pos + ivec2(i*4+1, 0), 0)[head],
                        texelFetch(attWeights, pos + ivec2(i*4+2, 0), 0)[head],
                        texelFetch(attWeights, pos + ivec2(i*4+3, 0), 0)[head]);
        weights[i] = wgt;
    }
    for (int i=numpacks; i < MATRIX_WEIGHTS; i++) weights[i] = vec4(0.0f);
}

void main(void) {
    float fuzz  = 0.25 * float(gl_VertexID & 1) / float(viewport.x);     // see "diamond-exit" rule in GL spec section 3.4.1
    gl_Position = vec4(attributes0.x + fuzz, 0.0f, 0.0f, 1.0f);
    highp int head = int(attributes0.y);
    highp int currenttoken = gl_InstanceID * MATRIX_WEIGHTS * 4;
    highp int numweights = max(0, min(tokenIdx+1 - currenttoken, MATRIX_WEIGHTS*4));
    fetchWeights(ivec2(currenttoken, head / 4), numweights, head & 3);
    weightData.x = numweights;
    weightData.y = currenttoken;
    valPos = (attributes0.x + 1.0) * 0.5 * float(viewport.x);
}

