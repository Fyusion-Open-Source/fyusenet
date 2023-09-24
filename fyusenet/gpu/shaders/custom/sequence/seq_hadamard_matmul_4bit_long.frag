/* -------------------------------------------------------------------------------------------------
 * Matrix-Multiplication + Hadamard Product for Sequences (long, 4-bit quant)  (c) Martin Wawro 2023
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
layout(binding=0) uniform sampler2D inputLayer0;
layout(binding=1) uniform sampler2D inputLayer1;
#ifdef USE_BIAS
layout(binding=5) uniform sampler2D biasData;
#endif
#ifdef USE_RESIDUAL
layout(binding=6) uniform sampler2D residual;
#endif
#else
uniform sampler2D inputLayer0;
uniform sampler2D inputLayer1;
#ifdef USE_BIAS
uniform sampler2D biasData;
#endif
#ifdef USE_RESIDUAL
uniform sampler2D residual;
#endif
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 inputPos;
flat in highp int colOffset;
flat in vec4 weights[MATRIX_WEIGHTS*NUM_LANES];
flat in int instanceMod;

#if NUM_LANES == 1
const vec4 masks[4] = vec4[4](vec4(1.0, 0.0, 0.0, 0.0),
                              vec4(0.0, 1.0, 0.0, 0.0),
                              vec4(0.0, 0.0, 1.0, 0.0),
                              vec4(0.0, 0.0, 0.0, 1.0));
#endif

#if NUM_LANES == 2
const mat2x4 masks[2] = mat2x4[2](
            mat2x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0)),
            mat2x4(vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)));
#endif

#include "shaders/activation.inc"

vec4 fetch(in ivec2 pos) {
#if (ACTIVATION_MASK & 3) == 0
     vec4 src = texelFetch(inputLayer0, pos, 0) * texelFetch(inputLayer1, pos, 0);
#endif
#if (ACTIVATION_MASK & 3) == 1
     vec4 src = activate(texelFetch(inputLayer0, pos, 0)) * texelFetch(inputLayer1, pos, 0);
#endif
#if (ACTIVATION_MASK & 3) == 2
     vec4 src = texelFetch(inputLayer0, pos, 0) * activate(texelFetch(inputLayer1, pos, 0));
#endif
#if (ACTIVATION_MASK & 3) == 3
     vec4 src = activate(texelFetch(inputLayer0, pos, 0)) * activate(texelFetch(inputLayer1, pos, 0));
#endif
    return src; 
}

void main(void) {
    highp ivec2 ipos = ivec2(colOffset, inputPos.y);
#if NUM_LANES == 1
    highp float accu = 0.0;
    for (int w=0; w < MATRIX_WEIGHTS; w++) {
        vec4 src = fetch(ipos+ivec2(w,0));
        accu += dot(src, weights[w]);
    }
    fragmentColor0 = masks[instanceMod] * accu;
#endif
#if NUM_LANES == 2
    highp vec2 accu = vec2(0.0);
    for (int w=0; w < MATRIX_WEIGHTS; w++) {
        vec4 src = fetch(ipos+ivec2(w,0));
        accu += vec2(dot(src, weights[w]), dot(src, weights[w+MATRIX_WEIGHTS]));
    }
    fragmentColor0 = masks[instanceMod] * accu;
#endif
#ifdef USE_BIAS
    fragmentColor0 += texelFetch(biasData, ivec2(inputPos.x, 0), 0);
#endif
#ifdef USE_RESIDUAL
    fragmentColor0 += texelFetch(residual, ivec2(inputPos.xy), 0);
#endif
}
