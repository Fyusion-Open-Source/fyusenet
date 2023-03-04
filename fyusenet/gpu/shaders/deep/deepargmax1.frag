/* ----------------------------------------------------------------------------
 * ArgMax Shader (Deep)                    Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
#define HIGH_PRECISION
#endif

#include "shaders/deep/fragpreamble.inc"

uniform highp float alpha1;
uniform highp float alpha2;
uniform highp float alpha3;

flat in highp ivec4 channelOffset;
flat in lowp ivec4 mask;      // mask to enable/disable channels in a 4-channel pixel

#include "shaders/activation.inc"

const highp vec4 maskout = vec4(FLT_MIN);

uniform highp ivec4 bitmask;


void main(void) {
  highp vec4 pix = activate(texture(inputLayer0,texCoord));
  fragmentColor0.z = pix.x;
  pix = mix(maskout, pix, equal(mask, ivec4(1)));
  highp ivec4 ipix = floatBitsToInt(pix);
  highp ivec4 masked = ipix & bitmask;
  highp vec4 completed = intBitsToFloat(masked | (channelOffset << PLACEMENT_BITS));
  highp vec2 mb = max(completed.xy, completed.zw);
  highp vec2 ma = max(pix.xy, pix.zw);
  fragmentColor0.xy = vec2(max(ma.x, ma.y), max(mb.x, mb.y));
  fragmentColor0.w = pix.x;
}
