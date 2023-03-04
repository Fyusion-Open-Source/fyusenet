/* ----------------------------------------------------------------------------
 * Transpose Conv 2x2 (shallow, vanilla)   Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer;
#else
uniform sampler2D inputLayer;
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

const mediump vec4 ones=vec4(1,1,1,1);

in highp vec2 texCoord;

uniform vec2 texStep;

uniform mat4 coeffs[CONVSIZE*NUM_LANES];

#ifdef POST_BATCHNORM
uniform vec4 batchnorm[NUM_LANES];
#endif

#include "shaders/activation.inc"

vec4 multiply(in vec4 inpix,in int offset, const in int bnscale) {
#ifdef POST_BATCHNORM
  return coeffs[offset]*inpix*batchnorm[bnscale];
#else
  return coeffs[offset]*inpix;
#endif
}

void procAndSet(in vec4 inpix,in int offset) {
  vec4 pix = activate(inpix);
#ifdef USE_BIAS
  fragmentColor0 = bias[0]+multiply(pix,offset, 0);
#if NUM_LANES > 1
  fragmentColor1 = bias[1]+multiply(pix,offset+CONVSIZE, 1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = bias[2]+multiply(pix,offset+2*CONVSIZE, 2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = bias[3]+multiply(pix,offset+3*CONVSIZE, 3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = bias[4]+multiply(pix,offset+4*CONVSIZE, 4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = bias[5]+multiply(pix,offset+5*CONVSIZE, 5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = bias[6]+multiply(pix,offset+6*CONVSIZE, 6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = bias[7]+multiply(pix,offset+7*CONVSIZE, 7);
#endif
#else
  fragmentColor0 = multiply(pix,offset, 0);
#if NUM_LANES > 1
  fragmentColor1 = multiply(pix,offset+CONVSIZE, 1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = multiply(pix,offset+2*CONVSIZE, 2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = multiply(pix,offset+3*CONVSIZE, 3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = multiply(pix,offset+4*CONVSIZE, 4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = multiply(pix,offset+5*CONVSIZE, 5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = multiply(pix,offset+6*CONVSIZE, 6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = multiply(pix,offset+7*CONVSIZE, 7);
#endif
#endif
}


void main(void) {  
#if STEP == 1
  procAndSet(texture(inputLayer,texCoord),0);
#endif
#if STEP == 2
  procAndSet(texture(inputLayer,texCoord-vec2(texStep.x,0)),0);
#endif
#if STEP == 3
  procAndSet(texture(inputLayer,texCoord+vec2(0,texStep.y)),0);
#endif
#if STEP == 4
  procAndSet(texture(inputLayer,texCoord+texStep),0);
#endif
}
