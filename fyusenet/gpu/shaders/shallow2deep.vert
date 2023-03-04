/* ----------------------------------------------------------------------------
 * Vertex Shader (Shallow -> Deep)         Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec4 attributes0;
in highp int attributes1;

out highp vec4 texCoord;
flat out int useTexUnit;

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = vec4(attributes0.z,attributes0.w,0.0,0.0);
  useTexUnit = attributes1;
}
