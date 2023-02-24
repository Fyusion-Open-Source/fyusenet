/* ----------------------------------------------------------------------------
 * TransConv Shader (Deep Tensor Format)   Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/convheader.inc"

uniform highp vec2 texStep;
uniform mediump int pass;

#ifdef NO_HALF
// requires 6 varyings in total (w/ residual)
flat in vec4 layer0coeffs[4];
#else
// requires 4 varyings in total (w/ residual)
flat in highp uvec4 layer0coeffs[2];
#endif

#include "shaders/activation.inc"
#include "shaders/deep/batchnorm.inc"
#include "shaders/deep/residual.inc"
#include "shaders/deep/computeconv.inc"
  
void main(void) {
  vec2 tmul = vec2(float(pass & 1),float((pass & 2)>>1));
  fragmentColor0 = compute(texture(inputLayer0,texCoord.xy-texStep*tmul),0);
#ifdef POST_BATCHNORM
  fragmentColor0 = applyBN(fragmentColor0, biasTexture, ivec4(texCoord.zw,0,1));
#else
  fragmentColor0 += texelFetch(biasTexture,ivec2(int(texCoord.z),0),0);
#endif
#ifdef USE_RESIDUAL
  fragmentColor0 += residual(residualLayer0,resCoord);
#endif
}
