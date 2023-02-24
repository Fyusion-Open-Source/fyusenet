/* ----------------------------------------------------------------------------
 * Default Convolution Vertex Shader       Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision mediump float;
precision highp int;

in highp vec4 attributes0;

out highp vec2 texCoord;

#ifdef USE_RESIDUAL
in highp vec2 attributes1;
out highp vec2 resCoord;
#endif

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = attributes0.zw;
#ifdef USE_RESIDUAL
  resCoord = attributes1.xy;
#endif
}

