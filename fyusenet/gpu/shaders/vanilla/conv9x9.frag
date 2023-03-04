/* ----------------------------------------------------------------------------
 * 9x9 Shallow Convolution (Vanilla)       Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/vanilla/conv_common.inc"
#include "shaders/activation.inc"
#include "shaders/vanilla/conv.inc"
#include "shaders/vanilla/residual.inc"

void main(void) {
#ifdef USE_BIAS
  fragmentColor0=bias[0];
#else
  fragmentColor0=vec4(0);
#endif
#if NUM_LANES > 1
#ifdef USE_BIAS
  fragmentColor1=bias[1];
#else
  fragmentColor1=vec4(0);
#endif
#endif
#if NUM_LANES > 2
#ifdef USE_BIAS
  fragmentColor2=bias[2];
#else
  fragmentColor2=vec4(0);
#endif
#endif
#if NUM_LANES > 3
#ifdef USE_BIAS
  fragmentColor3=bias[3];
#else
  fragmentColor3=vec4(0);
#endif
#endif
  // FIXME (mw) large dilation steps !
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2(-4*DILATION,0)),0);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2(-3*DILATION,0)),1);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2(-2*DILATION,0)),2);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2(-DILATION,0)),3);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( 0,0)),4);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( DILATION,0)),5);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( 2*DILATION,0)),6);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( 3*DILATION,0)),7);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( 4*DILATION,0)),8);
#ifdef USE_RESIDUAL
  if (addResidual>0) handleResidual();
#endif  // USE_RESIDUAL
}
