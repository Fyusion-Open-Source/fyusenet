/* ----------------------------------------------------------------------------
 * ArgMax Vertex Shader (Deep)             Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec4 attributes0;
in highp ivec4 attributes1;
in lowp ivec4 attributes2;

out highp vec2 texCoord;
flat out highp ivec4 channelOffset;
flat out lowp ivec4 mask; // mask to enable/disable channels in a 4-channel pixel

void main(void) {
  gl_Position = vec4(attributes0.x, attributes0.y, 0.0, 1.0);
  channelOffset = attributes1;
  texCoord = vec2(attributes0.z,attributes0.w);
  mask = attributes2;
}
