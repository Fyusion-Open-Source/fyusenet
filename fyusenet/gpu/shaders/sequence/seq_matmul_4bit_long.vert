/* -------------------------------------------------------------------------------------------------
 * Matrix-Multiplication for Sequences (long)                                  (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#define ZERO_WIDTH 8

#ifndef INSTANCE_OFFSET
#define INSTANCE_OFFSET 0
#endif

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
layout(binding=2) uniform highp usampler2D matrix;
layout(binding=3) uniform highp sampler2D scaleData;
layout(binding=4) uniform highp usampler2D zeroData;
#else
uniform highp usampler2D matrix;
uniform highp sampler2D scaleData;
uniform highp usampler2D zeroData;
#endif

in highp uint attributes0;

flat out vec4 weights[MATRIX_WEIGHTS*NUM_LANES];
flat out int instanceMod;
flat out highp int colOffset;
out highp vec2 inputPos;

uniform ivec2 viewport;        // viewport
uniform int quantGroupSize;    // quantization group size

void unpackMatrix(in ivec2 pos,
                  out vec4 wgt0, out vec4 wgt1, out vec4 wgt2, out vec4 wgt3,
                  out vec4 wgt4, out vec4 wgt5, out vec4 wgt6, out vec4 wgt7) {

    highp ivec2 spos = ivec2(pos.y, (pos.x * MATRIX_WEIGHTS * 4) / quantGroupSize);
    highp ivec2 zpos = ivec2(pos.y / ZERO_WIDTH, spos.y);
    highp uvec4 block = texelFetch(matrix, pos, 0);
    highp float scale = texelFetch(scaleData, spos, 0).r;
    highp uint zeroblock = texelFetch(zeroData, zpos, 0).r;
    int zmod = (pos.y % ZERO_WIDTH) * 4;
    highp float zero = scale * float(((int(zeroblock) >> zmod) & 0xF) + 1);
    // unpack 32 4-bit values into 8 vec4s
    wgt0 = scale * (vec4(block.x & 0xFu, (block.x >> 4) & 0xFu,  (block.x >> 8) & 0xFu, (block.x >> 12) & 0xFu)) - vec4(zero);
    wgt1 = scale * (vec4((block.x >> 16) & 0xFu, (block.x >> 20) & 0xFu, (block.x >> 24) & 0xFu, (block.x >> 28) & 0xFu)) - vec4(zero);
    wgt2 = scale * (vec4(block.y & 0xFu, (block.y >> 4) & 0xFu,(block.y >> 8) & 0xFu, (block.y >> 12) & 0xFu)) - vec4(zero);
    wgt3 = scale * (vec4((block.y >> 16) & 0xFu, (block.y >> 20) & 0xFu, (block.y >> 24) & 0xFu, (block.y >> 28) & 0xFu)) - vec4(zero);
    wgt4 = scale * (vec4(block.z & 0xFu, (block.z >> 4) & 0xFu,(block.z >> 8) & 0xFu, (block.z >> 12) & 0xFu)) - vec4(zero);
    wgt5 = scale * (vec4((block.z >> 16) & 0xFu, (block.z >> 20) & 0xFu, (block.z >> 24) & 0xFu, (block.z >> 28) & 0xFu)) - vec4(zero);
    wgt6 = scale * (vec4(block.w & 0xFu, (block.w >> 4) & 0xFu,(block.w >> 8) & 0xFu, (block.w >> 12) & 0xFu)) - vec4(zero);
    wgt7 = scale * (vec4((block.w >> 16) & 0xFu, (block.w >> 20) & 0xFu, (block.w >> 24) & 0xFu, (block.w >> 28) & 0xFu)) - vec4(zero);
}

// 4 instances to complete one column (RGBA),
void main(void) {
    highp int column = int(attributes0 >> 16);
    highp float row = float(attributes0 & 0xFFFFu) * float(viewport.y);
    highp float yfuzz = ((attributes0 & 0x1u) != 0u) ? 0.25 : 0.0;
    highp vec2 cr = (vec2(column, row) + vec2(0.25, yfuzz))  / vec2(viewport.xy);  // see "diamond-exit" rule in GL spec section 3.4.1
    gl_Position = vec4((cr * 2.0) - vec2(1.0), 0.0, 1.0);

#if NUM_LANES == 1
    instanceMod = (gl_InstanceID + INSTANCE_OFFSET) % 4;
    colOffset = ((gl_InstanceID + INSTANCE_OFFSET) / 4) * MATRIX_WEIGHTS;
    // compute column/row position in quantized matrix
    highp int weightcol = column*4 + instanceMod;                     // column in weight matrix, mtx itself is stored in CM order, so that actually is the y-coordinate of the texture
    highp int weightrow = ((gl_InstanceID + INSTANCE_OFFSET) / 4);    // row in weight matrix (compensated for packing), actually consistutes the x-coordinate of the texture (32x packing applied internally)
    inputPos = vec2(column, row);
#endif
#if NUM_LANES == 2
    instanceMod = (gl_InstanceID + INSTANCE_OFFSET) % 2;
    colOffset = ((gl_InstanceID + INSTANCE_OFFSET) / 2) * MATRIX_WEIGHTS;
    highp int weightcol = column*4 + 2*instanceMod;
    highp int weightrow = ((gl_InstanceID + INSTANCE_OFFSET) / 2);
    inputPos = vec2(column, row);
#endif

#if NUM_LANES >= 1
    // NOTE (mw) assuming column-major order on the weights
    unpackMatrix(ivec2(weightrow, weightcol),
                 weights[0], weights[1], weights[2], weights[3],
                 weights[4], weights[5], weights[6], weights[7]);
#endif
#if NUM_LANES == 2
    // NOTE (mw) assuming column-major order on the weights
    unpackMatrix(ivec2(weightrow, weightcol+1),
                 weights[8], weights[9], weights[10], weights[11],
                 weights[12], weights[13], weights[14], weights[15]);
#endif
}

