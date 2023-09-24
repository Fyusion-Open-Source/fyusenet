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
layout(binding=1) uniform sampler2D weights;
#else
uniform sampler2D inputLayer0;
uniform sampler2D weights;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 texCoord;
flat in highp float scale;

uniform highp int row;

void main(void) {
    vec4 val = texelFetch(inputLayer0, ivec2(texCoord.x, row), 0);
    vec4 wgt = texelFetch(weights, ivec2(texCoord.x, 0), 0);
    fragmentColor0 = wgt * val * scale;
}
