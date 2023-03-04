/* ----------------------------------------------------------------------------
 * Concatenation Shader (Vanilla)          Copyright (c) 2016-2022 Fyusion Inc.
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
layout(binding=1) uniform sampler2D inputLayer1;
layout(binding=2) uniform sampler2D inputLayer2;
layout(binding=3) uniform sampler2D inputLayer3;
layout(binding=4) uniform sampler2D inputLayer4;
layout(binding=5) uniform sampler2D inputLayer5;
layout(binding=6) uniform sampler2D inputLayer6;
layout(binding=7) uniform sampler2D inputLayer7;
#else
uniform sampler2D inputLayer0;
uniform sampler2D inputLayer1;
uniform sampler2D inputLayer2;
uniform sampler2D inputLayer3;
uniform sampler2D inputLayer4;
uniform sampler2D inputLayer5;
uniform sampler2D inputLayer6;
uniform sampler2D inputLayer7;
#endif

layout(location=0) out vec4 fragmentColor0;
#if NUM_LANES > 1
layout(location=1) out vec4 fragmentColor1;
#endif
#if NUM_LANES > 2
layout(location=2) out vec4 fragmentColor2;
#endif
#if NUM_LANES > 3
layout(location=3) out vec4 fragmentColor3;
#endif
#if NUM_LANES > 4
layout(location=4) out vec4 fragmentColor4;
#endif
#if NUM_LANES > 5
layout(location=5) out vec4 fragmentColor5;
#endif
#if NUM_LANES > 6
layout(location=6) out vec4 fragmentColor6;
#endif
#if NUM_LANES > 7
layout(location=7) out vec4 fragmentColor7;
#endif

in highp vec2 texCoord;
#ifdef USE_RESIDUAL
in highp vec2 resCoord;
#endif

#include "shaders/activation.inc"

vec4 fetch(in sampler2D inLayer,in vec2 tc) {
#if SHIFT==0
  return activate(texture(inLayer,tc));
#endif
#if SHIFT==1
  return activate(texture(inLayer,tc).gbar);
#endif
#if SHIFT==2
  return activate(texture(inLayer,tc).barg);
#endif
#if SHIFT==3
  return activate(texture(inLayer,tc).argb);
#endif
}

void main(void) {
#if TRAIL==4
  vec4 inpix = activate(texture(inputLayer0,texCoord));
  fragmentColor0 = inpix;
#else
  vec4 inpix0 = fetch(inputLayer0,texCoord);
  vec4 inpix1 = activate(texture(inputLayer1,texCoord));
#if TRAIL==3
  fragmentColor0.rgb = inpix0.rgb;
  fragmentColor0.a = inpix1.r;
#endif
#if TRAIL==2
  fragmentColor0.rg = inpix0.rg;
  fragmentColor0.ba = inpix1.rg;
#endif
#if TRAIL==1
  fragmentColor0.r = inpix0.r;
  fragmentColor0.gba = inpix1.rgb;
#endif
#endif
}
