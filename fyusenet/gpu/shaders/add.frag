/* ----------------------------------------------------------------------------
 * Addition Layer                          Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#else
precision highp float;
precision highp int;
precision highp sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D op1Layer0;
layout(binding=1) uniform sampler2D op2Layer0;
layout(binding=2) uniform sampler2D op1Layer1;
layout(binding=3) uniform sampler2D op2Layer1;
layout(binding=4) uniform sampler2D op1Layer2;
layout(binding=5) uniform sampler2D op2Layer2;
layout(binding=6) uniform sampler2D op1Layer3;
layout(binding=7) uniform sampler2D op2Layer3;
#if NUM_LANES > 4
layout(binding=8) uniform sampler2D op1Layer4;
layout(binding=9) uniform sampler2D op2Layer4;
layout(binding=10) uniform sampler2D op1Layer5;
layout(binding=11) uniform sampler2D op2Layer5;
layout(binding=12) uniform sampler2D op1Layer6;
layout(binding=13) uniform sampler2D op2Layer6;
layout(binding=14) uniform sampler2D op1Layer7;
layout(binding=15) uniform sampler2D op2Layer7;
#endif
#else
uniform sampler2D op1Layer0;
uniform sampler2D op2Layer0;
uniform sampler2D op1Layer1;
uniform sampler2D op2Layer1;
uniform sampler2D op1Layer2;
uniform sampler2D op2Layer2;
uniform sampler2D op1Layer3;
uniform sampler2D op2Layer3;
#if NUM_LANES > 4
uniform sampler2D op1Layer4;
uniform sampler2D op2Layer4;
uniform sampler2D op1Layer5;
uniform sampler2D op2Layer5;
uniform sampler2D op1Layer6;
uniform sampler2D op2Layer6;
uniform sampler2D op1Layer7;
uniform sampler2D op2Layer7;
#endif
#endif

in highp vec2 texCoord;

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

#include "shaders/activation.inc"

vec4 fetch(in sampler2D sampler) {
  return activate(texture(sampler,texCoord));
}

void main(void) {
#if SIGNED == 1
  fragmentColor0 = fetch(op1Layer0)-fetch(op2Layer0);
#if NUM_LANES > 1
  fragmentColor1 = fetch(op1Layer1)-fetch(op2Layer1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = fetch(op1Layer2)-fetch(op2Layer2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = fetch(op1Layer3)-fetch(op2Layer3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = fetch(op1Layer4)-fetch(op2Layer4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = fetch(op1Layer5)-fetch(op2Layer5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = fetch(op1Layer6)-fetch(op2Layer6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = fetch(op1Layer7)-fetch(op2Layer7);
#endif
#else  // positive (addition)
  fragmentColor0 = fetch(op1Layer0)+fetch(op2Layer0);
#if NUM_LANES > 1
  fragmentColor1 = fetch(op1Layer1)+fetch(op2Layer1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = fetch(op1Layer2)+fetch(op2Layer2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = fetch(op1Layer3)+fetch(op2Layer3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = fetch(op1Layer4)+fetch(op2Layer4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = fetch(op1Layer5)+fetch(op2Layer5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = fetch(op1Layer6)+fetch(op2Layer6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = fetch(op1Layer7)+fetch(op2Layer7);
#endif
#endif
}
