/* ----------------------------------------------------------------------------
 * 3x3 conv vertex shader (deep-format)     Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=4) uniform highp sampler2D inputDisplacements;
#ifdef NO_HALF
layout(binding=5) uniform sampler2D inputCoeffs;
#else
layout(binding=5) uniform highp usampler2D inputCoeffs;
#endif
#else
uniform highp sampler2D inputDisplacements;
uniform highp usampler2D inputCoeffs;
#endif

in highp vec4 attributes0;
in highp ivec2 attributes1;
in highp vec2 attributes2;

out highp vec4 texCoord;
#ifdef NO_HALF
flat out mediump vec4 layer0coeffs[12];
#else
// requires 8 varyings in total (w/ residual)
flat out highp uvec4 layer0coeffs[6];
#endif

#ifdef USE_RESIDUAL
out highp vec2 resCoord;
#endif

uniform int numInputTiles;

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = vec4(attributes0.z,attributes0.w,0.0,0.0);
#ifdef INSTANCE_OFFSET
  int instance = gl_InstanceID + INSTANCE_OFFSET;
#else
  int instance = gl_InstanceID;
#endif
  int windowindex = instance / numInputTiles;
  int intile = instance % numInputTiles;
  texCoord.xy += texelFetch(inputDisplacements,ivec2(intile,windowindex),0).rg;
#ifdef NO_HALF
  intile *= 4*KERNEL;
#else
  intile *= 2*KERNEL;
#endif
  int ybase = windowindex + attributes1.x;
  // fetch weights
#ifdef NO_HALF
  for (int i=0;i<12;i++) {
    layer0coeffs[i] = texelFetch(inputCoeffs,ivec2(intile+i,ybase),0);
  }
#else
  layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(intile,ybase),0);
  layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(intile+1,ybase),0);
  layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(intile+2,ybase),0);
  layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(intile+3,ybase),0);
  layer0coeffs[4] = texelFetch(inputCoeffs,ivec2(intile+4,ybase),0);
  layer0coeffs[5] = texelFetch(inputCoeffs,ivec2(intile+5,ybase),0);
#endif
#if !defined(NO_BIAS) || defined(POST_BATCHNORM)
  if (instance == 0) {
    texCoord.z = float(attributes1.y+1);
  } else {
    texCoord.z = 0.0;
  }
#else
  texCoord.z = 0.0;
#endif
  texCoord.w = float(attributes1.y+1);
#ifdef USE_RESIDUAL
  resCoord = attributes2.xy;
#endif
}

