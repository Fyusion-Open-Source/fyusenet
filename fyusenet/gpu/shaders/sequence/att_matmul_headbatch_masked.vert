/* -------------------------------------------------------------------------------------------------
 * Attention-Layer Attention Weights / Value Multiplication (Batched)          (c) Martin Wawro 2023
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

in highp uint attributes0;

#ifdef BINDING_SUPPORT
layout(binding=1) uniform sampler2D attWeights;
#else
uniform sampler2D attWeights;
#endif

flat out vec4 weights[MATRIX_WEIGHTS];
flat out highp ivec2 weightData;        // x: #weights to use, y: row-offset in value data
out highp vec2 valPos;                  // x: column in value data, y: subscript

uniform highp ivec2 viewport;           // x,y: viewport size (x: 4 heads, y: query length)
uniform highp ivec4 tileParams;         // x: value x offset, y: value column span per line primitive, z: y-offset for weights, w: tokenIndex

void fetchWeights(in ivec2 pos, in int nweights) {
    for (int i=0; i < nweights; i++) weights[i] = texelFetch(attWeights, pos+ivec2(i,0), 0);
    for (int i=nweights; i < MATRIX_WEIGHTS; i++) weights[i] = vec4(0.0);
}

void main(void) {
    highp int tokenidx =  int(attributes0>>16);
    highp int row = tokenidx - tileParams.w;
    int right = int(attributes0 & 0x1u);
    highp int dup = int((attributes0>>1) & 0x7FFFu);
    highp float nrow = ((float(row) + 0.25) / float(viewport.y)) * 2.0 - 1.0;       // see "diamond-exit" rule in GL spec section 3.4.1
    float fuzz = (right > 0) ? 0.25/float(viewport.x) : 0.0;
    gl_Position = vec4(-1.0 + float(2*right)+fuzz, nrow, 0.0, 1.0);                 // see "diamond-exit" rule in GL spec section 3.4.1
    highp int numweights = max(0, min(tokenidx+1 - dup*MATRIX_WEIGHTS, MATRIX_WEIGHTS));
    fetchWeights(ivec2(dup * MATRIX_WEIGHTS, row + tileParams.z), numweights);
    weightData.x = numweights;
    weightData.y = dup * MATRIX_WEIGHTS;
    valPos.x = float(tileParams.x + right * tileParams.y);
    valPos.y = float(right) * 4.0;
}

