/* ----------------------------------------------------------------------------
 * Concatenation Vertex Shader (Vanilla)   Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision mediump float;
precision highp int;

in highp vec4 attributes0;

out highp vec2 texCoord;

#ifdef INPUT_TRANSFORM
uniform mat4 texTrans;
#endif

void main(void) {
    gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
#ifdef INPUT_TRANSFORM
    vec4 tmp = texTrans*vec4(attributes0.zw,0.0,1.0);
    texCoord = tmp.xy;
#else
    texCoord = attributes0.zw;
#endif
}
