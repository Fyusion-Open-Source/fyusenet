/* -------------------------------------------------------------------------------------------------
 * Matrix-Multiplication for Sequences (short)                                 (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#define ZERO_WIDTH 8

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
layout(binding=2) uniform highp usampler2D matrix;
layout(binding=3) uniform highp sampler2D scaleData;
layout(binding=4) uniform highp usampler2D zeroData;
#ifdef USE_BIAS
layout(binding=5) uniform sampler2D biasData;
#endif
#ifdef USE_RESIDUAL
layout(binding=6) uniform sampler2D residual;
#endif
#else
uniform sampler2D inputLayer0;
uniform highp usampler2D matrix;
uniform highp sampler2D scaleData;
uniform highp usampler2D zeroData;
#ifdef USE_BIAS
uniform sampler2D biasData;
#endif
#ifdef USE_RESIDUAL
uniform sampler2D residual;
#endif
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 inputPos;
flat in highp ivec2 colOffset;
uniform highp int quantGroupSize;

#include "shaders/activation.inc"

void unpackMatrix(in ivec2 pos, out vec4 wgt[MATRIX_WEIGHTS*4]) {
    // x: row, y: column
    // matrix is stored column-major, pos.x provides the "row" index
    highp ivec2 spos = ivec2(pos.y, (pos.x * MATRIX_WEIGHTS * 4) / quantGroupSize);
    for (int i=0; i < 4; i++) {     // loop over 4 columns (stored as rows in the col-major matrix)
        highp uvec4 block = texelFetch(matrix, pos + ivec2(0,i), 0);
        highp ivec2 zpos = ivec2((pos.y+i) / ZERO_WIDTH, spos.y);
        highp float scale = texelFetch(scaleData, spos + ivec2(i,0), 0).r;
        highp uint zeroblock = texelFetch(zeroData, zpos, 0).r;
        int zmod = ((pos.y+i) % ZERO_WIDTH) * 4;
        highp float zero = scale * float(((int(zeroblock) >> zmod) & 0xF) + 1);
        wgt[i*MATRIX_WEIGHTS+0] = scale * (vec4(block.x & 0xFu, (block.x >> 4) & 0xFu,  (block.x >> 8) & 0xFu, (block.x >> 12) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+1] = scale * (vec4((block.x >> 16) & 0xFu, (block.x >> 20) & 0xFu, (block.x >> 24) & 0xFu, (block.x >> 28) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+2] = scale * (vec4(block.y & 0xFu, (block.y >> 4) & 0xFu,(block.y >> 8) & 0xFu, (block.y >> 12) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+3] = scale * (vec4((block.y >> 16) & 0xFu, (block.y >> 20) & 0xFu, (block.y >> 24) & 0xFu, (block.y >> 28) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+4] = scale * (vec4(block.z & 0xFu, (block.z >> 4) & 0xFu,(block.z >> 8) & 0xFu, (block.z >> 12) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+5] = scale * (vec4((block.z >> 16) & 0xFu, (block.z >> 20) & 0xFu, (block.z >> 24) & 0xFu, (block.z >> 28) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+6] = scale * (vec4(block.w & 0xFu, (block.w >> 4) & 0xFu,(block.w >> 8) & 0xFu, (block.w >> 12) & 0xFu)) - vec4(zero);
        wgt[i*MATRIX_WEIGHTS+7] = scale * (vec4((block.w >> 16) & 0xFu, (block.w >> 20) & 0xFu, (block.w >> 24) & 0xFu, (block.w >> 28) & 0xFu)) - vec4(zero);
    }
}


void main(void) {
    vec4 weights[MATRIX_WEIGHTS*4];
    highp vec4 accu = vec4(0.0);
    highp ivec2 ipos = ivec2(colOffset.x, inputPos.y);
    for (int pack=0; pack < MATRIX_PACKS; pack++) {
        // NOTE (mw) weight matrix is in column-major order
        //                     matrix row         matrix column
        unpackMatrix(ivec2(colOffset.y + pack, int(inputPos.x) * 4), weights);
        // Multiply MATRIX_WEIGHTS*4 rows and 4 columns of the matrix with the input vector
        for (int w=0; w < MATRIX_WEIGHTS; w++) {
            vec4 wgt0 = weights[w];
            vec4 wgt1 = weights[w+MATRIX_WEIGHTS];
            vec4 wgt2 = weights[w+2*MATRIX_WEIGHTS];
            vec4 wgt3 = weights[w+3*MATRIX_WEIGHTS];
            vec4 src = activate(texelFetch(inputLayer0, ipos+ivec2(w, 0), 0));
            accu += vec4(dot(src, wgt0), dot(src, wgt1), dot(src, wgt2), dot(src, wgt3));
        }
    }
#ifdef USE_BIAS
    accu += texelFetch(biasData, ivec2(inputPos.x, 0), 0);
#endif
#ifdef USE_RESIDUAL
    fragmentColor0 = accu + texelFetch(residual, ivec2(inputPos.xy), 0);
#else
    fragmentColor0 = accu;
#endif
}
