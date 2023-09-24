/* -------------------------------------------------------------------------------------------------
 * Attention-Layer Attention Weights / Value Multiplication                    (c) Martin Wawro 2023
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
#else
uniform sampler2D inputLayer0;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 valPos;                  // x: column in value matrix ; y: subscript to use
flat in vec4 weights[MATRIX_WEIGHTS];  // attention weights taking part in the matrix mult
flat in highp ivec2 weightData;        // x: #weights to use, y: row-offset in value data

void main(void) {
    highp int xpos = int(valPos.x);
    int subscript = int(valPos.y);
    highp vec4 accu = vec4(0.0);
    for (int i=0; i < weightData.x; i++) {
        vec4 val = texelFetch(inputLayer0, ivec2(xpos, weightData.y + i), 0);
        accu += weights[i][subscript] * val;
    }
    fragmentColor0 = accu;
}

