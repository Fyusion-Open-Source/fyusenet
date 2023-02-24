/* ----------------------------------------------------------------------------
 * ImgPatch Vertex Shader (Deep)            Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec2 attributes0;
in highp ivec4 attributes1;

flat out highp ivec4 positions;

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  positions = attributes1;
}

