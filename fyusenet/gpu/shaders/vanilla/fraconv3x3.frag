/* ----------------------------------------------------------------------------
 * 3x3 Fractional Convolution (Shallow)    Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/vanilla/conv_common.inc"

#include "shaders/activation.inc"
#include "shaders/vanilla/fractional.inc"
#include "shaders/vanilla/residual.inc"

void main(void) {
  vec4 inpix = texture(inputLayer,texCoord-vec2(2.0*texStep,0));
  process(inpix);
  inpix = texture(inputLayer,texCoord-vec2(texStep,0));
  processAndAdd(inpix,1);
  inpix = texture(inputLayer,texCoord);
  processAndAdd(inpix,2);
#ifdef USE_RESIDUAL
  if (addResidual>0) handleResidual();
#endif  // USE_RESIDUAL
}
