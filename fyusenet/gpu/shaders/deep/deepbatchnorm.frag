/* ----------------------------------------------------------------------------
 * BatchNorm Shader (Deep Tensor Format)    Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

#ifdef BINDING_SUPPORT
layout(binding=1) uniform sampler2D residualLayer0;
#else
uniform sampler2D residualLayer0;
#endif

#ifdef USE_RESIDUAL
in highp vec2 resCoord;
#endif

flat in highp vec4 scales;
flat in highp vec4 biases;

#include "shaders/activation.inc"

void main(void) {
  fragmentColor0 = activate(texture(inputLayer0, texCoord.xy)) * scales + biases;
#ifdef USE_RESIDUAL
  vec4 residual = texture(residualLayer0, resCoord.xy);
#ifdef RELU_ON_RESIDUAL
  residual = max(vec4(0.0),residual);
#endif
#ifdef BATCHNORM_ON_RESIDUAL
  residual = residual * scales + biases;
#endif
  fragmentColor0 += residual;
#endif
}
