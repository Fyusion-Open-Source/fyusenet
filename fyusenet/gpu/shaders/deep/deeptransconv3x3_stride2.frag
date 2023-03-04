/* ----------------------------------------------------------------------------
 * TransConv Shader (Deep Format Tensor)    Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/convheader.inc"

uniform mediump int pass;
uniform vec4 texStep;

// requires 10 varyings in total (w/ residual)
flat in highp vec2 texClamp;
flat in highp uvec4 layer0coeffs[8];

#include "shaders/activation.inc"
#include "shaders/deep/batchnorm.inc"
#include "shaders/deep/residual.inc"
#include "shaders/deep/computeconv.inc"

vec4 clampedTexture(in sampler2D sampler, in vec2 tc) {
  float mr = float(all(lessThanEqual(tc,texClamp+texStep.zw)));
  float ml = float(all(greaterThanEqual(tc,texClamp)));
  return texture(sampler,tc)*ml*mr;
}

void main(void) {
  vec2 tc = texCoord.xy;
  if (pass == 0) {
    fragmentColor0 = compute(clampedTexture(inputLayer0,tc),0);
    fragmentColor0 += compute(clampedTexture(inputLayer0,tc-vec2(2.0*texStep.x,0)),2);
    fragmentColor0 += compute(clampedTexture(inputLayer0,tc-vec2(0.0,2.0*texStep.y)),4);
    fragmentColor0 += compute(clampedTexture(inputLayer0,tc-vec2(2.0*texStep.x,2.0*texStep.y)),6);
  } else if (pass == 1) {
    fragmentColor0 =  compute(clampedTexture(inputLayer0,tc-vec2(texStep.x,0)),0);
    fragmentColor0 += compute(clampedTexture(inputLayer0,tc-vec2(texStep.x,2.0*texStep.y)),2);
  } else if (pass == 2) {
    fragmentColor0 =  compute(clampedTexture(inputLayer0,tc-vec2(0,texStep.y)),0);
    fragmentColor0 += compute(clampedTexture(inputLayer0,tc-vec2(2.0*texStep.x,texStep.y)),2);
  } else if (pass == 3) {
    fragmentColor0 =  compute(clampedTexture(inputLayer0,tc-texStep.xy),0);
  }
#ifdef POST_BATCHNORM
  fragmentColor0 = applyBN(fragmentColor0, biasTexture, ivec4(texCoord.zw,0,1));
#else
  fragmentColor0 += texelFetch(biasTexture,ivec2(int(texCoord.z),0),0);
#endif
#ifdef USE_RESIDUAL
  fragmentColor0 += residual(residualLayer0,clampAndRes.zw, biasTexture, texCoord.zw);
#endif
}
