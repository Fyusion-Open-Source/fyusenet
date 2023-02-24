/* ----------------------------------------------------------------------------
 * Concatenation Shader (Deep Tensor Fmt)  Copyright (c) 2016-2022 Fyusion Inc.
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
#else
uniform sampler2D inputLayer0;
uniform sampler2D inputLayer1;
uniform sampler2D inputLayer2;
uniform sampler2D inputLayer3;
#endif

layout(location=0) out vec4 fragmentColor;

in highp vec4 texCoord0;
in highp vec4 texCoord1;

uniform int numTextures;

flat in ivec4 texComponents;
flat in ivec4 texShift;

#include "shaders/activation.inc"

vec4 fetch(in sampler2D inTex,in vec2 tc) {
  return activate(texture(inTex,tc));
}

void main(void) {
  if (numTextures == 1) {
    if (texShift.x == 0) fragmentColor = fetch(inputLayer0,texCoord0.xy);
    else {
      vec4 pixel = fetch(inputLayer0, texCoord0.xy);
      for (int i=0; i < texComponents.x;i++) fragmentColor[i] = pixel[i+texShift.x];
    }
  } else if (numTextures == 2) {
    int fi = 0;
    vec4 pixel = fetch(inputLayer0, texCoord0.xy);
    for (int i=0; i < texComponents.x;i++) fragmentColor[fi++] = pixel[i+texShift.x];
    pixel = fetch(inputLayer1, texCoord0.zw);
    for (int i=0; i < texComponents.y;i++) fragmentColor[fi++] = pixel[i+texShift.y];
  } else if (numTextures == 3) {
    int fi = 0;
    vec4 pixel = fetch(inputLayer0,texCoord0.xy);
    for (int i=0; i < texComponents.x;i++) fragmentColor[fi++] = pixel[i+texShift.x];
    pixel = fetch(inputLayer1, texCoord0.zw);
    for (int i=0; i < texComponents.y;i++) fragmentColor[fi++] = pixel[i+texShift.y];
    pixel = fetch(inputLayer2, texCoord1.xy);
    for (int i=0; i < texComponents.z;i++) fragmentColor[fi++] = pixel[i+texShift.z];
  } else if (numTextures == 4) {
    vec4 pixel = fetch(inputLayer0,texCoord0.xy);
    fragmentColor.r = pixel[texShift.x];
    pixel = fetch(inputLayer1, texCoord0.zw);
    fragmentColor.g = pixel.r;
    pixel = fetch(inputLayer2, texCoord1.xy);
    fragmentColor.b = pixel.r;
    pixel = fetch(inputLayer3, texCoord1.zw);
    fragmentColor.a = pixel.r;
  }
}
