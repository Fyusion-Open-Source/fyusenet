/* -------------------------------------------------------------------------------------------------
 * Attention-Layer QK Dot-Product (single query token)                         (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
#ifdef HIGH_PRECISION
precision highp sampler2D;
#else
precision mediump sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
layout(binding=1) uniform sampler2D inputLayer1;
#else
uniform sampler2D inputLayer0;
uniform sampler2D inputLayer1;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 keyHeadPos;
flat in int innerBatch;

uniform highp ivec4 inputParams;     // x: head-size (pixels), y: #heads (pixels), z: #key tokens, w: #query tokens
uniform highp float scaling;         // scaling value for "scaled" dot-product

void main(void) {
    int headidx = int(keyHeadPos.y) * 4;
    int keyidx = int(keyHeadPos.x);
    ivec2 lhspos = ivec2(headidx * inputParams.x + innerBatch * INNER_BATCH_SIZE, 0);
    ivec2 rhspos = ivec2(headidx * inputParams.x + innerBatch * INNER_BATCH_SIZE, keyidx);
    vec4 accu = vec4(0.0);
    for (int b=0; b < INNER_BATCH_SIZE; b++) {
        highp vec4 lhs0 = texelFetch(inputLayer0, lhspos, 0);
        highp vec4 lhs1 = texelFetch(inputLayer0, lhspos+ivec2(inputParams.x,0), 0);
        highp vec4 lhs2 = texelFetch(inputLayer0, lhspos+ivec2(2*inputParams.x,0), 0);
        highp vec4 lhs3 = texelFetch(inputLayer0, lhspos+ivec2(3*inputParams.x,0), 0);
        highp vec4 rhs0 = texelFetch(inputLayer1, rhspos, 0);
        highp vec4 rhs1 = texelFetch(inputLayer1, rhspos+ivec2(inputParams.x,0), 0);
        highp vec4 rhs2 = texelFetch(inputLayer1, rhspos+ivec2(2*inputParams.x,0), 0);
        highp vec4 rhs3 = texelFetch(inputLayer1, rhspos+ivec2(3*inputParams.x,0), 0);
        accu += vec4(dot(lhs0, rhs0), dot(lhs1, rhs1), dot(lhs2, rhs2), dot(lhs3, rhs3));
        lhspos.x++;
        rhspos.x++;
    }
    fragmentColor0 = accu * scaling;
}
