/* ----------------------------------------------------------------------------
 * 5x5 conv for deep-format tensors        Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/convheader.inc"

#ifdef NO_HALF
// requires 22 varyings in total (w/ residual)
flat in vec4 layer0coeffs[20];
#else
// requires 12 varyings in total (w/ residual)
flat in highp uvec4 layer0coeffs[10];
#endif

#include "shaders/activation.inc"
#include "shaders/deep/batchnorm.inc"
#include "shaders/deep/computeconv.inc"
#include "shaders/deep/residual.inc"

#ifdef LARGE_DILATION
uniform highp float dilationStep;
#endif

void main(void) {
#ifdef LARGE_DILATION
    fragmentColor0 =  compute(texture(inputLayer0,texCoord.xy-vec2(2*dilationStep,0)),0);
    fragmentColor0 += compute(texture(inputLayer0,texCoord.xy-vec2(dilationStep,0)),2);
    fragmentColor0 += compute(texture(inputLayer0,texCoord.xy),4);
    fragmentColor0 += compute(texture(inputLayer0,texCoord.xy+vec2(dilationStep,0)),6);
    fragmentColor0 += compute(texture(inputLayer0,texCoord.xy+vec2(2*dilationStep,0)),8);
#else
    fragmentColor0 =  compute(textureOffset(inputLayer0,texCoord.xy,ivec2(-2*DILATION,0)),0);
    fragmentColor0 += compute(textureOffset(inputLayer0,texCoord.xy,ivec2(-DILATION,0)),2);
    fragmentColor0 += compute(textureOffset(inputLayer0,texCoord.xy,ivec2( 0,0)),4);
    fragmentColor0 += compute(textureOffset(inputLayer0,texCoord.xy,ivec2( DILATION,0)),6);
    fragmentColor0 += compute(textureOffset(inputLayer0,texCoord.xy,ivec2(2*DILATION,0)),8);
#endif
#if !defined(NO_BIAS) || defined(POST_BATCHNORM)
#ifdef POST_BATCHNORM
    fragmentColor0 = applyBN(fragmentColor0, biasTexture, ivec4(texCoord.zw,0,1));
#else
    fragmentColor0 += texelFetch(biasTexture,ivec2(int(texCoord.z),0),0);
#endif
#endif
#ifdef USE_RESIDUAL
    fragmentColor0 += residual(residualLayer0, resCoord.xy, biasTexture, texCoord.zw);
#endif
}
