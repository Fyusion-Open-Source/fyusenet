/* ----------------------------------------------------------------------------
 * Singleton Arithmetic Shader (Deep)      Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

#include "shaders/activation.inc"

uniform highp float operand;

void main(void) {
    vec4 op = vec4(operand);
    vec4 data = activate(texture(inputLayer0,texCoord.xy));
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
    fragmentColor0 = data;
}
