 /* ----------------------------------------------------------------------------
 * TransConv Shader (Deep)                  Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifdef BINDING_SUPPORT
layout(binding=4) uniform highp sampler2D inputDisplacements;
#ifdef NO_HALF
layout(binding=5) uniform sampler2D inputCoeffs;
#else
layout(binding=5) uniform highp usampler2D inputCoeffs;
#endif
#else
uniform highp sampler2D inputDisplacements;
#ifdef NO_HALF
uniform sampler2D inputCoeffs;
#else
uniform highp usampler2D inputCoeffs;
#endif
#endif

in highp vec4 attributes0;
in highp ivec2 attributes1;
in highp vec2 attributes2;

out highp vec4 texCoord;
#ifdef NO_HALF
flat out vec4 layer0coeffs[4];
#else
// requires 4 varyings in total (w/ residual)
flat out highp uvec4 layer0coeffs[2];
#endif

#ifdef USE_RESIDUAL
out highp vec2 resCoord;
#endif

uniform int numInputTiles;
uniform mediump int pass;

#ifdef NO_HALF
#define TSTEP 4
#else
#define TSTEP 2
#endif

void main(void) {
#ifdef INSTANCE_OFFSET
  int instance = gl_InstanceID + INSTANCE_OFFSET;
#else
  int instance = gl_InstanceID;
#endif
  int intile = instance % numInputTiles;
  texCoord = vec4(attributes0.z,attributes0.w,0.0,0.0);
  texCoord.xy += texelFetch(inputDisplacements,ivec2(intile,0),0).rg;
#ifdef NO_HALF
  intile *= 4*KERNEL;
#else
  intile *= 2*KERNEL;
#endif
  int ybase = KERNEL * attributes1.x;
  if (pass == 0) {
    layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(intile,ybase),0);
    layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(intile+1,ybase),0);
#ifdef NO_HALF
    layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(intile+2,ybase),0);
    layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(intile+3,ybase),0);
#endif
  } else if (pass == 1) {
    layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(intile+TSTEP,ybase),0);
    layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(intile+TSTEP+1,ybase),0);
#ifdef NO_HALF
    layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(intile+TSTEP+2,ybase),0);
    layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(intile+TSTEP+3,ybase),0);
#endif
  } else if (pass == 2) {
    layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(intile,ybase+1),0);
    layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(intile+1,ybase+1),0);
#ifdef NO_HALF
    layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(intile+2,ybase+1),0);
    layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(intile+3,ybase+1),0);
#endif
  } else {
    layer0coeffs[0] = texelFetch(inputCoeffs,ivec2(intile+TSTEP,ybase+1),0);
    layer0coeffs[1] = texelFetch(inputCoeffs,ivec2(intile+TSTEP+1,ybase+1),0);
#ifdef NO_HALF
    layer0coeffs[2] = texelFetch(inputCoeffs,ivec2(intile+TSTEP+2,ybase+1),0);
    layer0coeffs[3] = texelFetch(inputCoeffs,ivec2(intile+TSTEP+3,ybase+1),0);
#endif
  }
  gl_Position = vec4(attributes0.x,attributes0.y,0,1.0);
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

