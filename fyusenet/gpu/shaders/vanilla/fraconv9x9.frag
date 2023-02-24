/* ----------------------------------------------------------------------------
 * 9x9 Fractional Convolution (Shallow)    Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/vanilla/conv_common.inc"

#include "shaders/activation.inc"
#include "shaders/vanilla/fractional.inc"
#include "shaders/vanilla/residual.inc"

void main(void) {
  vec4 inpix = texture(inputLayer,texCoord-vec2(4.0*texStep,0));
  process(inpix);
  inpix = texture(inputLayer,texCoord-vec2(3.0*texStep,0));
  processAndAdd(inpix,1);
  inpix = texture(inputLayer,texCoord-vec2(2.0*texStep,0));
  processAndAdd(inpix,2);
  inpix = texture(inputLayer,texCoord-vec2(texStep,0));
  processAndAdd(inpix,3);
  inpix = texture(inputLayer,texCoord);
  processAndAdd(inpix,4);
  inpix = texture(inputLayer,texCoord+vec2(texStep,0));
  processAndAdd(inpix,5);
  inpix = texture(inputLayer,texCoord+vec2(2.0*texStep,0));
  processAndAdd(inpix,6);
  inpix = texture(inputLayer,texCoord+vec2(3.0*texStep,0));
  processAndAdd(inpix,7);
  inpix = texture(inputLayer,texCoord+vec2(4.0*texStep,0));
  processAndAdd(inpix,8);
#ifdef USE_RESIDUAL
  if (addResidual>0) handleResidual();
#endif  // USE_RESIDUAL
}
