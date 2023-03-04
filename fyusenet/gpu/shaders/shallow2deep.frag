/* ----------------------------------------------------------------------------
 * Order Conversion (Shallow -> Deep)      Copyright (c) 2016-2022 Fyusion Inc.
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

in highp vec4 texCoord;
flat in int useTexUnit;

#include "shaders/activation.inc"

vec4 fetch(in sampler2D texUnit,in vec2 tc) {
  return activate(texture(texUnit,tc));
}

void main(void) {
  vec2 tc = texCoord.xy;  
  if (useTexUnit < 4) {
    if (useTexUnit >=2) {
      if (useTexUnit == 2) fragmentColor0 = fetch(inputLayer2,tc);
      else fragmentColor0 = fetch(inputLayer3,tc);
    } else {
      if (useTexUnit == 0) fragmentColor0 = fetch(inputLayer0,tc);
      else fragmentColor0 = fetch(inputLayer1,tc);
    }
  } else {
    if (useTexUnit >= 6) {
      if (useTexUnit == 6) fragmentColor0 = fetch(inputLayer6,tc);
      else fragmentColor0 = fetch(inputLayer7,tc);
    } else {
      if (useTexUnit == 4) fragmentColor0 = fetch(inputLayer4,tc);
      else fragmentColor0 = fetch(inputLayer5,tc);
    }
  }
}
