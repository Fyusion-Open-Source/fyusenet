/* -------------------------------------------------------------------------------------------------
 * Matrix-Multiplication for Sequences (long)                                  (c) Martin Wawro 2023
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
#ifdef USE_BIAS
layout(binding=5) uniform sampler2D biasData;
#endif
#ifdef USE_RESIDUAL
layout(binding=6) uniform sampler2D residual;
#endif
#else
uniform sampler2D inputLayer0;
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
flat in highp uvec4 weights[MATRIX_WEIGHTS*(NUM_LANES/2)];
flat in int instanceMod;

#if NUM_LANES == 2
const mat2x4 masks[2] = mat2x4[2](
            mat2x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0)),
            mat2x4(vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)));
#endif

#include "shaders/activation.inc"

void main(void) {
    highp ivec2 ipos = ivec2(colOffset, inputPos.y);
#if NUM_LANES == 2
    highp vec2 accu2 = vec2(0.0);
    for (int x=0,w=0; x < MATRIX_WEIGHTS; x+=2, w++) {
        vec4 srcA = activate(texelFetch(inputLayer0, ipos+ivec2(x,0), 0));
        vec4 srcB = activate(texelFetch(inputLayer0, ipos+ivec2(x+1,0), 0));
        vec4 wgtA = vec4(unpackHalf2x16(weights[w].x),unpackHalf2x16(weights[w].y));
        vec4 wgtB = vec4(unpackHalf2x16(weights[w].z),unpackHalf2x16(weights[w].w));
        vec4 wgtC = vec4(unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].x),unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].y));
        vec4 wgtD = vec4(unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].z),unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].w));
        accu2 += vec2(dot(srcA,wgtA) + dot(srcB,wgtB), dot(srcA,wgtC) + dot(srcB,wgtD));
    }
    fragmentColor0 = masks[instanceMod] * accu2;
#endif
#if NUM_LANES == 4
    highp vec4 accu4 = vec4(0.0);
    for (int x=0,w=0; x < MATRIX_WEIGHTS; x+=2, w++) {
        vec4 srcA = activate(texelFetch(inputLayer0, ipos+ivec2(x,0), 0));
        vec4 srcB = activate(texelFetch(inputLayer0, ipos+ivec2(x+1,0), 0));
        vec4 wgtA = vec4(unpackHalf2x16(weights[w].x),unpackHalf2x16(weights[w].y));
        vec4 wgtB = vec4(unpackHalf2x16(weights[w].z),unpackHalf2x16(weights[w].w));
        vec4 wgtC = vec4(unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].x),unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].y));
        vec4 wgtD = vec4(unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].z),unpackHalf2x16(weights[w+MATRIX_WEIGHTS/2].w));
        vec4 wgtE = vec4(unpackHalf2x16(weights[w+MATRIX_WEIGHTS].x),unpackHalf2x16(weights[w+MATRIX_WEIGHTS].y));
        vec4 wgtF = vec4(unpackHalf2x16(weights[w+MATRIX_WEIGHTS].z),unpackHalf2x16(weights[w+MATRIX_WEIGHTS].w));
        vec4 wgtG = vec4(unpackHalf2x16(weights[w+3*MATRIX_WEIGHTS/2].x),unpackHalf2x16(weights[w+3*MATRIX_WEIGHTS/2].y));
        vec4 wgtH = vec4(unpackHalf2x16(weights[w+3*MATRIX_WEIGHTS/2].z),unpackHalf2x16(weights[w+3*MATRIX_WEIGHTS/2].w));
        accu4 += vec4(dot(srcA,wgtA) + dot(srcB,wgtB), dot(srcA,wgtC) + dot(srcB,wgtD),
                     dot(srcA,wgtE) + dot(srcB,wgtF), dot(srcA,wgtG) + dot(srcB,wgtH));
    }
    fragmentColor0 = accu4;
#endif
#ifdef USE_BIAS
    fragmentColor0 += texelFetch(biasData, ivec2(inputPos.x, 0), 0);
#endif
#ifdef USE_RESIDUAL
    fragmentColor0 += texelFetch(residual, ivec2(inputPos.xy), 0);
#endif
}
