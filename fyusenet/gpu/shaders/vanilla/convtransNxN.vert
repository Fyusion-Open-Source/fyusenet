/* ----------------------------------------------------------------------------
 * Transpose Convolution Vertex Shader     Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

in highp vec3 attributes0;
in highp vec2 attributes1;

out highp vec2 texCoord;

#ifdef USE_RESIDUAL
out highp vec2 resCoord;
#endif

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,attributes0.z,1.0);
  texCoord = attributes1.xy;
#ifdef USE_RESIDUAL
  resCoord = attributes1.xy;
#endif
}

