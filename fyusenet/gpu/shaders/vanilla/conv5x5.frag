/* ----------------------------------------------------------------------------
 * 5x5 Shallow Convolution (Vanilla)       Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/vanilla/conv_common.inc"
#include "shaders/activation.inc"
#include "shaders/vanilla/conv.inc"
#include "shaders/vanilla/residual.inc"

// FIXME (mw) large dilation steps !

void main(void) {
  procAndSet(textureOffset(inputLayer,texCoord,ivec2(-2*DILATION,0)),0);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2(-DILATION,0)),1);
  procAndAdd(texture(inputLayer,texCoord),2);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( DILATION,0)),3);
  procAndAdd(textureOffset(inputLayer,texCoord,ivec2( 2*DILATION,0)),4);
#ifdef USE_RESIDUAL
  if (addResidual>0) handleResidual();
#endif  // USE_RESIDUAL
}
