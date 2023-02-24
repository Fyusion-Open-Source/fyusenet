/* ----------------------------------------------------------------------------
 * Scaling Layer Neural Network Shader     Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifdef HIGH_PRECISION
precision highp float;
precision highp int;
precision highp sampler2D;
#else
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
#if NUM_LANES > 1
layout(binding=1) uniform sampler2D inputLayer1;
#endif
#if NUM_LANES > 2
layout(binding=2) uniform sampler2D inputLayer2;
#endif
#if NUM_LANES > 3
layout(binding=3) uniform sampler2D inputLayer3;
#endif
#if NUM_LANES > 4
layout(binding=4) uniform sampler2D inputLayer4;
#endif
#if NUM_LANES > 5
layout(binding=5) uniform sampler2D inputLayer5;
#endif
#if NUM_LANES > 6
layout(binding=6) uniform sampler2D inputLayer6;
#endif
#if NUM_LANES > 7
layout(binding=7) uniform sampler2D inputLayer7;
#endif
#else
uniform sampler2D inputLayer0;
#if NUM_LANES > 1
uniform sampler2D inputLayer1;
#endif
#if NUM_LANES > 2
uniform sampler2D inputLayer2;
#endif
#if NUM_LANES > 3
uniform sampler2D inputLayer3;
#endif
#if NUM_LANES > 4
uniform sampler2D inputLayer4;
#endif
#if NUM_LANES > 5
uniform sampler2D inputLayer5;
#endif
#if NUM_LANES > 6
uniform sampler2D inputLayer6;
#endif
#if NUM_LANES > 7
uniform sampler2D inputLayer7;
#endif
#endif


layout(location=0) out vec4 fragmentColor0;
#if NUM_LANES > 1
layout(location=1) out vec4 fragmentColor1;
#endif
#if NUM_LANES > 2
layout(location=2) out vec4 fragmentColor2;
#endif
#if NUM_LANES > 3
layout(location=3) out vec4 fragmentColor3;
#endif
#if NUM_LANES > 4
layout(location=4) out vec4 fragmentColor4;
#endif
#if NUM_LANES > 5
layout(location=5) out vec4 fragmentColor5;
#endif
#if NUM_LANES > 6
layout(location=6) out vec4 fragmentColor6;
#endif
#if NUM_LANES > 7
layout(location=7) out vec4 fragmentColor7;
#endif


in highp vec2 texCoord;

#include "shaders/activation.inc"

void main(void) {
  fragmentColor0 = activate(texture(inputLayer0,texCoord));
#if NUM_LANES > 1
  fragmentColor1 = activate(texture(inputLayer1,texCoord));
#endif
#if NUM_LANES > 2
  fragmentColor2 = activate(texture(inputLayer2,texCoord));
#endif
#if NUM_LANES > 3
  fragmentColor3 = activate(texture(inputLayer3,texCoord));
#endif
#if NUM_LANES > 4
  fragmentColor4 = activate(texture(inputLayer4,texCoord));
#endif
#if NUM_LANES > 5
  fragmentColor5 = activate(texture(inputLayer5,texCoord));
#endif
#if NUM_LANES > 6
  fragmentColor6 = activate(texture(inputLayer6,texCoord));
#endif
#if NUM_LANES > 7
  fragmentColor7 = activate(texture(inputLayer7,texCoord));
#endif
}
