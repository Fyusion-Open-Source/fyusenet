/* ----------------------------------------------------------------------------
 * Depthwise Conv Shader (Deep)             Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/convheader.inc"

#ifdef NO_HALF
// requires 13 varyings in total (w/ residual)
flat in vec4 layer0coeffs[12];
#else
// requires 8 varyings in total (w/ residual)
flat in highp uvec4 layer0coeffs[6];
#endif

#include "shaders/deep/batchnorm.inc"
#include "shaders/deep/residual.inc"
#include "shaders/activation.inc"

#ifndef NO_HALF
vec4 compute(in vec4 tex00,in vec4 tex01,in vec4 tex02,in int offset) {
  highp uvec4 w = layer0coeffs[offset];
  vec4 r = tex00 * vec4(unpackHalf2x16(w.x),unpackHalf2x16(w.y));
  r += tex01 * vec4(unpackHalf2x16(w.z),unpackHalf2x16(w.w));
  w = layer0coeffs[offset+1];
  r += tex02 * vec4(unpackHalf2x16(w.x),unpackHalf2x16(w.y));
  return r;
}
#else
vec4 compute(in vec4 tex00,in vec4 tex01,in vec4 tex02,in int offset) {
  return tex00 * layer0coeffs[2*offset] + tex01 * layer0coeffs[2*offset+1] + tex02 * layer0coeffs[2*offset+2];
}
#endif

void main(void) {
  vec4 t0 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2(-DILATION,-DILATION)));
  vec4 t1 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2(        0,-DILATION)));
  vec4 t2 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2( DILATION,-DILATION)));
  fragmentColor0 = compute(t0,t1,t2,0);
  t0 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2(-DILATION,0)));
  t1 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2(        0,0)));
  t2 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2( DILATION,0)));
  fragmentColor0 += compute(t0,t1,t2,2);
  t0 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2(-DILATION, DILATION)));
  t1 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2(        0, DILATION)));
  t2 = activate(textureOffset(inputLayer0,texCoord.xy,ivec2( DILATION, DILATION)));
  fragmentColor0 += compute(t0,t1,t2,4);
#if !defined(NO_BIAS) || defined(POST_BATCHNORM)
#ifdef POST_BATCHNORM
  fragmentColor0 = applyBN(fragmentColor0, biasTexture, ivec4(texCoord.zw,0,1));
#else
  fragmentColor0 += texelFetch(biasTexture,ivec2(int(texCoord.z),0),0);
#endif
#endif
#ifdef USE_RESIDUAL
  fragmentColor0 += residual(residualLayer0,resCoord);
#endif
}
