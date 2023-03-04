/* ----------------------------------------------------------------------------
 * NxN partial conv fragment shader (deep-fmt)  Copyright (c) 2023 Fyusion Inc.
 * Even sized parts
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/convheader.inc"

#ifdef NO_HALF
flat in vec4 layer0coeffs[COEFF_VARYINGS];
#else
flat in highp uvec4 layer0coeffs[COEFF_VARYINGS];
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
    fragmentColor0 =  compute(texture(inputLayer0,texCoord), OFFSET0);
#if NET_KERNEL >= 6
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy+vec2(-3*dilationStep,0)),OFFSET6a);
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy+vec2( 2*dilationStep,0)),OFFSET4b);
#endif
#if NET_KERNEL >= 4
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy+vec2(-2*dilationStep,0)),OFFSET4a);
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy+vec2( dilationStep,0)),OFFSET2b);
#endif
#if NET_KERNEL >= 2
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy+vec2(-dilationStep,0)), OFFSET2a);
#endif
#else
    fragmentColor0 = compute(textureOffset(inputLayer0,texCoord.xy, ivec2(0,0)), OFFSET0);
#if NET_KERNEL >= 6
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy,ivec2(-3*DILATION,0)),OFFSET6a);
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy,ivec2( 2*DILATION,0)),OFFSET4b);
#endif
#if NET_KERNEL >= 4
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy,ivec2(-2*DILATION,0)),OFFSET4a);
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy,ivec2( DILATION,0)),OFFSET2b);
#endif
#if NET_KERNEL >= 2
    fragmentColor0 +=  compute(textureOffset(inputLayer0,texCoord.xy,ivec2(-DILATION,0)), OFFSET2a);
#endif
#endif
#if !defined(NO_BIAS) || defined(POST_BATCHNORM)
#ifdef POST_BATCHNORM
    fragmentColor0 = applyBN(fragmentColor0, biasTexture, ivec4(texCoord.zw,0,1));
#else
    fragmentColor0 += texelFetch(biasTexture,ivec2(int(texCoord.z),0),0);
#endif
#endif
#ifdef USE_RESIDUAL
    fragmentColor0 += residual(residualLayer0,resCoord,biasTexture, texCoord.zw);
#endif
}
