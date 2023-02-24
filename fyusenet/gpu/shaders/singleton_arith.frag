/* ----------------------------------------------------------------------------
 * Singleton Arithmetic Shader             Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/funcpreamble.inc"

uniform highp float operand;

#include "shaders/activation.inc"


vec4 process(in sampler2D sampler,in vec4 op,const in int bidx) {
    vec4 data = activate(texture(sampler, texCoord));
#ifdef ARITH_OP_ADD
    data += op;
#endif
#ifdef ARITH_OP_SUB
    data -= op;
#endif
#ifdef ARITH_OP_DIV
    data /= op;
#endif
#ifdef ARITH_OP_MUL
    data *= op;
#endif
#ifdef POST_BATCHNORM
    data = data*batchnorm[bidx]+batchnorm[bidx+1];
#endif
    return data;
}


void main(void) {
    vec4 op = vec4(operand);
    fragmentColor0 = process(inputLayer0, op, 0);
#if NUM_LANES > 1
    fragmentColor1 = process(inputLayer1, op, 2);
#endif
#if NUM_LANES > 2
    fragmentColor2 = process(inputLayer2, op, 4);
#endif
#if NUM_LANES > 3
    fragmentColor3 = process(inputLayer3, op, 6);
#endif
#if NUM_LANES > 4
    fragmentColor4 = process(inputLayer4, op, 8);
#endif
#if NUM_LANES > 5
    fragmentColor5 = process(inputLayer5, op, 10);
#endif
#if NUM_LANES > 6
    fragmentColor6 = process(inputLayer6, op, 12);
#endif
#if NUM_LANES > 7
    fragmentColor7 = process(inputLayer7, op, 14);
#endif
}
