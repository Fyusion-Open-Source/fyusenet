/* ----------------------------------------------------------------------------
 * Depthwise 3x3 conv (deep-format)        Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
#ifdef NO_HALF
layout(binding=WEIGHT_UNIT) uniform sampler2D inputCoeffs;
#else
layout(binding=WEIGHT_UNIT) uniform highp usampler2D inputCoeffs;
#endif
#else
#ifdef NO_HALF
uniform sampler2D inputCoeffs;
#else
uniform highp usampler2D inputCoeffs;
#endif
#endif

in highp vec4 attributes0;
in highp ivec3 attributes1;
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

void main(void) {
  gl_Position = vec4(attributes0.x,attributes0.y,0.0,1.0);
  texCoord = vec4(attributes0.z,attributes0.w,0.0,0.0);
#ifdef NO_HALF
  int xbase = attributes1.x * 4;
#else
  int xbase = attributes1.x * 2;
#endif
  int ybase = attributes1.z * KERNEL;
  // fetch weights
#ifdef NO_HALF
  layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(xbase,ybase),0);
  layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(xbase+1,ybase),0);
  layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(xbase+2,ybase),0);
  layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(xbase+3,ybase),0);
  layer0coeffs[4] = texelFetch(inputCoeffs,ivec2(xbase,ybase+1),0);
  layer0coeffs[5] = texelFetch(inputCoeffs,ivec2(xbase+1,ybase+1),0);
  layer0coeffs[6] = texelFetch(inputCoeffs,ivec2(xbase+2,ybase+1),0);
  layer0coeffs[7] = texelFetch(inputCoeffs,ivec2(xbase+3,ybase+1),0);
  layer0coeffs[8] = texelFetch(inputCoeffs,ivec2(xbase,ybase+2),0);
  layer0coeffs[9] = texelFetch(inputCoeffs,ivec2(xbase+1,ybase+2),0);
  layer0coeffs[10] = texelFetch(inputCoeffs,ivec2(xbase+2,ybase+2),0);
  layer0coeffs[11] = texelFetch(inputCoeffs,ivec2(xbase+3,ybase+2),0);
#else
  layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(xbase,ybase),0);
  layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(xbase+1,ybase),0);
  layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(xbase,ybase+1),0);
  layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(xbase+1,ybase+1),0);
  layer0coeffs[4] = texelFetch(inputCoeffs,ivec2(xbase,ybase+2),0);
  layer0coeffs[5] = texelFetch(inputCoeffs,ivec2(xbase+1,ybase+2),0);
#endif
#if !defined(NO_BIAS) || defined(POST_BATCHNORM)
  texCoord.z = float(attributes1.y+1);
#else
  texCoord.z = 0.0;
#endif
  texCoord.w = float(attributes1.y+1);
#ifdef USE_RESIDUAL
  resCoord.xy = attributes2.xy;
#endif
}

