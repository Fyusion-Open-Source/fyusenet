/* ----------------------------------------------------------------------------
 * Default Vertex Shader                   Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec4 attributes0;
out highp vec2 texCoord;

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = attributes0.zw;
}

