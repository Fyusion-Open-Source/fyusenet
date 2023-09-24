/* -------------------------------------------------------------------------------------------------
 * RMSNormLayer for Sequence Layouts for single tokens                         (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
#ifndef HIGH_PRECISION
precision mediump sampler2D;
#else
precision highp sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
#else
uniform sampler2D inputLayer0;
#endif

in highp vec2 attributes0;

out highp vec2 texCoord;
flat out highp float scale;

uniform highp ivec2 embedWidth;         // x: pixels, y: elements
uniform highp int row;

void main(void) {
    float fuzz = (attributes0.y * 0.25) / float(embedWidth.x);
    gl_Position = vec4((attributes0.yx + vec2(fuzz, 0.5)) * 2.0 - vec2(1.0), 0.0, 1.0);  // see "diamond-exit" rule in GL spec section 3.4.1
    texCoord = vec2(embedWidth.x, 1) * attributes0.yx;
    highp vec4 accu = vec4(1.0);
    if (gl_VertexID == 1) {
        accu = vec4(0.0);
        for (int i=0; i < embedWidth.x; i++) {
            vec4 val = texelFetch(inputLayer0, ivec2(i, row), 0);
            accu += val * val;
        }
    }
    scale = inversesqrt(1.0e-6 + dot(accu, vec4(1.0)) / float(embedWidth.y));
}

