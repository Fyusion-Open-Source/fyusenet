/* ----------------------------------------------------------------------------
 * 3x3 Depthwise Conv  (Shallow Tensor)    Copyright (c) 2016-2022 Fyusion Inc.
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
uniform mediump sampler2D inputLayer0;
uniform mediump sampler2D inputLayer1;
uniform mediump sampler2D inputLayer2;
uniform mediump sampler2D inputLayer3;
uniform mediump sampler2D inputLayer4;
uniform mediump sampler2D inputLayer5;
uniform mediump sampler2D inputLayer6;
uniform mediump sampler2D inputLayer7;
#endif

#ifdef USE_RESIDUAL
#ifdef BINDING_SUPPORT
layout(binding=8) uniform sampler2D resLayer0;
layout(binding=9) uniform sampler2D resLayer1;
layout(binding=10) uniform sampler2D resLayer2;
layout(binding=11) uniform sampler2D resLayer3;
layout(binding=12) uniform sampler2D resLayer4;
layout(binding=13) uniform sampler2D resLayer5;
layout(binding=14) uniform sampler2D resLayer6;
layout(binding=15) uniform sampler2D resLayer7;
#else
uniform sampler2D resLayer0;
uniform sampler2D resLayer1;
uniform sampler2D resLayer2;
uniform sampler2D resLayer3;
uniform sampler2D resLayer4;
uniform sampler2D resLayer5;
uniform sampler2D resLayer6;
uniform sampler2D resLayer7;
#endif
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

#ifdef USE_RESIDUAL
uniform int addResidual;
#endif

uniform vec4 coeffs[CONVSIZE*CONVSIZE*NUM_INPUT_LANES*CHANNEL_MULTIPLIER];

#ifdef USE_BIAS
uniform vec4 bias[NUM_LANES];
#endif

#ifdef POST_BATCHNORM
uniform vec4 batchnorm[NUM_LANES];
#endif

#include "shaders/activation.inc"

vec4 conv3x3(in sampler2D tex,in vec2 tc,in int co) {
  vec4 pix,accu=vec4(0);
  pix = activate(textureOffset(tex,tc,ivec2(-1,-1)));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2( 0,-1)));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2( 1,-1)));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2(-1,0)));
  accu += pix*coeffs[co++];
  pix = activate(texture(tex,tc));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2( 1, 0)));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2(-1,1)));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2( 0,1)));
  accu += pix*coeffs[co++];
  pix = activate(textureOffset(tex,tc,ivec2( 1,1)));
  accu += pix*coeffs[co++];
  return accu;
}

void fetch(in sampler2D tex,in vec2 tc,in int co,inout vec4 frag[CHANNEL_MULTIPLIER]) {
  frag[0] = conv3x3(tex,texCoord,co);
}


#ifdef USE_RESIDUAL
vec4 fetchResidual(in sampler2D res,in vec2 rc) {
  vec4 pix = texture(res,rc);
#ifdef RELU_ON_RESIDUAL
  pix = max(vec4(0.0),pix);
#endif
  return pix;
}

void handleResidual() {
  fragmentColor0 += fetchResidual(resLayer0,resCoord);
#if NUM_LANES > 1
  fragmentColor1 += fetchResidual(resLayer1,resCoord);
#endif
#if NUM_LANES > 2
  fragmentColor2 += fetchResidual(resLayer2,resCoord);
#endif
#if NUM_LANES > 3
  fragmentColor3 += fetchResidual(resLayer3,resCoord);
#endif
#if NUM_LANES > 4
  fragmentColor4 += fetchResidual(resLayer4,resCoord);
#endif
#if NUM_LANES > 5
  fragmentColor5 += fetchResidual(resLayer5,resCoord);
#endif
#if NUM_LANES > 6
  fragmentColor6 += fetchResidual(resLayer6,resCoord);
#endif
#if NUM_LANES > 7
  fragmentColor7 += fetchResidual(resLayer7,resCoord);
#endif
}
#endif


void main(void) {
  vec4 frag[CHANNEL_MULTIPLIER];
  fetch(inputLayer0,texCoord,0,frag);
  fragmentColor0 = frag[0];  
#ifdef POST_BATCHNORM
  fragmentColor0 *= batchnorm[0];
#endif
#ifdef USE_BIAS
  fragmentColor0 += bias[0];
#endif

#if NUM_INPUT_LANES > 1
  fetch(inputLayer1,texCoord,CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor1 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor1 *= batchnorm[1];
#endif
#ifdef USE_BIAS
  fragmentColor1 += bias[1];
#endif
#endif

#if NUM_INPUT_LANES > 2
  fetch(inputLayer2,texCoord,2*CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor2 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor2 *= batchnorm[2];
#endif
#ifdef USE_BIAS
  fragmentColor2 += bias[2];
#endif
#endif

#if NUM_INPUT_LANES > 3
  fetch(inputLayer3,texCoord,3*CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor3 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor3 *= batchnorm[3];
#endif
#ifdef USE_BIAS
  fragmentColor3 += bias[3];
#endif
#endif

#if NUM_INPUT_LANES > 4
  fetch(inputLayer4,texCoord,4*CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor4 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor4 *= batchnorm[4];
#endif
#ifdef USE_BIAS
  fragmentColor4 += bias[4];
#endif
#endif

#if NUM_INPUT_LANES > 5
  fetch(inputLayer5,texCoord,5*CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor5 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor5 *= batchnorm[5];
#endif
#ifdef USE_BIAS
  fragmentColor5 += bias[5];
#endif
#endif

#if NUM_INPUT_LANES > 6
  fetch(inputLayer6,texCoord,6*CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor6 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor6 *= batchnorm[6];
#endif
#ifdef USE_BIAS
  fragmentColor6 += bias[6];
#endif
#endif

#if NUM_INPUT_LANES > 7
  fetch(inputLayer7,texCoord,7*CONVSIZE*CONVSIZE*CHANNEL_MULTIPLIER,frag);
  fragmentColor7 = frag[0];
#ifdef POST_BATCHNORM
  fragmentColor7 *= batchnorm[7];
#endif
#ifdef USE_BIAS
  fragmentColor7 += bias[7];
#endif
#endif

#ifdef USE_RESIDUAL
  handleResidual();
#endif

}
