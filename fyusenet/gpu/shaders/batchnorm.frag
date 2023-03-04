/* ----------------------------------------------------------------------------
 * BatchNorm Layer Shader                  Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision mediump float;
precision lowp int;
precision mediump sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
layout(binding=1) uniform sampler2D inputLayer1;
layout(binding=2) uniform sampler2D inputLayer2;
layout(binding=3) uniform sampler2D inputLayer3;
layout(binding=4) uniform sampler2D inputLayer4;
layout(binding=5) uniform sampler2D inputLayer5;
layout(binding=6) uniform sampler2D inputLayer6;
layout(binding=7) uniform sampler2D inputLayer7;
#else
uniform sampler2D inputLayer0;
uniform sampler2D inputLayer1;
uniform sampler2D inputLayer2;
uniform sampler2D inputLayer3;
uniform sampler2D inputLayer4;
uniform sampler2D inputLayer5;
uniform sampler2D inputLayer6;
uniform sampler2D inputLayer7;
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


uniform vec4 biasscale[NUM_LANES*2];


vec4 fetch(in sampler2D sampler,in int offset) {
  vec4 tex = texture(sampler,texCoord);
  return biasscale[offset]+tex*biasscale[offset+1];
}


void main(void) {
  fragmentColor0 = fetch(inputLayer0,0);
#if NUM_LANES > 1
  fragmentColor1 = fetch(inputLayer1,2);
#endif
#if NUM_LANES > 2
  fragmentColor2 = fetch(inputLayer2,4);
#endif
#if NUM_LANES > 3
  fragmentColor3 = fetch(inputLayer3,6);
#endif
#if NUM_LANES > 4
  fragmentColor4 = fetch(inputLayer4,8);
#endif
#if NUM_LANES > 5
  fragmentColor5 = fetch(inputLayer5,10);
#endif
#if NUM_LANES > 6
  fragmentColor6 = fetch(inputLayer6,12);
#endif
#if NUM_LANES > 7
  fragmentColor7 = fetch(inputLayer7,14);
#endif
}
