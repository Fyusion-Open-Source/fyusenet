/* ----------------------------------------------------------------------------
 * ImgPatch Shader (Deep)                  Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

// NOTE (mw) complex code due to precision issues when using texture interpolation

precision mediump float;
precision lowp int;
precision mediump sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer;
#else
uniform sampler2D inputLayer;
#endif

layout(location=0) out vec4 fragmentColor;

flat in highp ivec4 positions;

uniform int window;

#include "shaders/activation.inc"

vec4 fetch() {
  ivec2 mypos =ivec2(gl_FragCoord.xy-vec2(0.5,0.5));
  ivec2 relpos = (mypos-positions.xy)*window;
  ivec2 srcpos = (relpos+positions.zw);
  return texelFetch(inputLayer,srcpos,0);
}


void main(void) {
  fragmentColor = activate(fetch());
}
