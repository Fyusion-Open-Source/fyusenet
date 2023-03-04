/* ----------------------------------------------------------------------------
 * Concatenation Shader (deep format)      Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec2 posAttributes;
in highp vec4 texAttrs0;
in highp vec4 texAttrs1;
in highp ivec4 texCompAttrs;
in highp ivec4 texShiftAttrs;

out highp vec4 texCoord0;
out highp vec4 texCoord1;
flat out highp ivec4 texComponents;
flat out highp ivec4 texShift;


void main(void) {
  gl_Position = vec4(posAttributes.x, posAttributes.y, 0.0, 1.0);
  texCoord0 = texAttrs0;
  texCoord1 = texAttrs1;
  texComponents = texCompAttrs;
  texShift = texShiftAttrs;
}

