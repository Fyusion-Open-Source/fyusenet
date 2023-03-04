/* ----------------------------------------------------------------------------
 * BatchNorm Shader (Deep)                 Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

in highp vec4 attributes0;
in highp vec4 attributes1;
in highp vec4 attributes2;

out highp vec2 texCoord;

#ifdef USE_RESIDUAL
out highp vec2 resCoord;
#endif

flat out highp vec4 scales;
flat out highp vec4 biases;

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = attributes0.zw;
#ifdef USE_RESIDUAL
  resCoord = 0.5 * vec2(attributes0.x+1.0, attributes0.y+1.0);
#endif
  scales = attributes1;
  biases = attributes2;
}

