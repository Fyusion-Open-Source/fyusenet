/* ----------------------------------------------------------------------------
 * Default Vertex Shader (Deep)            Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec4 attributes0;
out highp vec2 texCoord;

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = vec2(attributes0.z,attributes0.w);
  gl_PointSize = 1.0;
}
