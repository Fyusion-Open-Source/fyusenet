/* ----------------------------------------------------------------------------
 * Vertex Shader         (Shallow -> Deep)  Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec2 posAttributes;

in highp vec4 attributes0;
out highp vec4 texCoord0;
#if NUM_MRT > 2
in highp vec4 attributes1;
out highp vec4 texCoord1;
#endif
#if NUM_MRT > 4
in highp vec4 attributes2;
out highp vec4 texCoord2;
#endif
#if NUM_MRT > 6
in highp vec4 attributes3;
out highp vec4 texCoord3;
#endif

void main(void) {
  gl_Position = vec4(posAttributes.x,posAttributes.y,0.0,1.0);
  texCoord0 = attributes0;
#if NUM_MRT > 2
  texCoord1 = attributes1;
#endif
#if NUM_MRT > 4
  texCoord2 = attributes2;
#endif
#if NUM_MRT > 6
  texCoord3 = attributes3;
#endif
}
