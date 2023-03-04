/* ----------------------------------------------------------------------------
 * Order conversion (Deep -> Shallow)      Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#else
precision highp float;
precision highp int;
precision highp sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
#else
uniform sampler2D inputLayer0;
#endif


layout(location=0) out vec4 fragmentColor0;
#if NUM_MRT > 1
layout(location=1) out vec4 fragmentColor1;
#endif
#if NUM_MRT > 2
layout(location=2) out vec4 fragmentColor2;
#endif
#if NUM_MRT > 3
layout(location=3) out vec4 fragmentColor3;
#endif
#if NUM_MRT > 4
layout(location=4) out vec4 fragmentColor4;
#endif
#if NUM_MRT > 5
layout(location=5) out vec4 fragmentColor5;
#endif
#if NUM_MRT > 6
layout(location=6) out vec4 fragmentColor6;
#endif
#if NUM_MRT > 7
layout(location=7) out vec4 fragmentColor7;
#endif

in highp vec4 texCoord0;
#if NUM_MRT > 2
in highp vec4 texCoord1;
#endif
#if NUM_MRT > 4
in highp vec4 texCoord2;
#endif
#if NUM_MRT > 6
in highp vec4 texCoord3;
#endif

#include "shaders/activation.inc"

vec4 fetch(in vec2 tc) {
  vec4 data = texture(inputLayer0,tc);
  return activate(data);
}

uniform int useMRT;

void main(void) {    
  fragmentColor0 = fetch(texCoord0.xy);
#if NUM_MRT > 1
  if (useMRT>1) fragmentColor1 = fetch(texCoord0.zw);
#endif
#if NUM_MRT > 2
  if (useMRT>2) fragmentColor2 = fetch(texCoord1.xy);
#endif
#if NUM_MRT > 3
  if (useMRT>3) fragmentColor3 = fetch(texCoord1.zw);
#endif
#if NUM_MRT > 4
  if (useMRT>4) fragmentColor4 = fetch(texCoord2.xy);
#endif
#if NUM_MRT > 5
  if (useMRT>5) fragmentColor5 = fetch(texCoord2.zw);
#endif
#if NUM_MRT > 6
  if (useMRT>6) fragmentColor6 = fetch(texCoord3.xy);
#endif
#if NUM_MRT > 7
  if (useMRT>7) fragmentColor7 = fetch(texCoord3.zw);
#endif
}
