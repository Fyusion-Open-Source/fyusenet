/* -------------------------------------------------------------------------------------------------
 * Projection shader for token scoring                                         (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump sampler2D;
#else
precision highp sampler2D;
#endif

precision highp float;
precision highp int;

in highp vec4 attributes0;

out highp vec2 outPosition;
out highp vec2 tokenXY;
flat out vec4 instanceData[INSTANCE_WIDTH];

uniform highp ivec2 viewport;
uniform highp int token;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputEmbeddings;
#else
uniform sampler2D inputEmbeddings;
#endif

void main(void) {
    gl_Position = vec4(attributes0.xy, 0.0, 1.0);
    outPosition = attributes0.zw * vec2(viewport);
    highp int tkx = gl_InstanceID * INSTANCE_WIDTH;
    tokenXY.x = float(tkx);
    tokenXY.y = float(token);
    for (int i=0; i < INSTANCE_WIDTH; i++) {
        instanceData[i] = texelFetch(inputEmbeddings, ivec2(tkx + i, token), 0);
    }
}

