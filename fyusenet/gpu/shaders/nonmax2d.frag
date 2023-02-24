/* ----------------------------------------------------------------------------
 * 2D Non-Maximum Suppression              Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision mediump float;
precision lowp int;
precision mediump sampler2D;


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

#define THRESH 0.1

vec4 process(in sampler2D sampler) {
  vec4 flags=vec4(0);
  vec4 mid =    texture(sampler,texCoord);
  vec4 right =  textureOffset(sampler,texCoord,ivec2(1,0));
  vec4 left =   textureOffset(sampler,texCoord,ivec2(-1,0));
  vec4 top =    textureOffset(sampler,texCoord,ivec2(0,-1));
  vec4 bottom = textureOffset(sampler,texCoord,ivec2(0,1));
  flags.r = (mid.r >= right.r && mid.r >= left.r && mid.r >= top.r && mid.r >= bottom.r && mid.r > THRESH) ? 1.0 : 0.0;
  flags.g = (mid.g >= right.g && mid.g >= left.g && mid.g >= top.g && mid.g >= bottom.g && mid.g > THRESH) ? 1.0 : 0.0;
  flags.b = (mid.b >= right.b && mid.b >= left.b && mid.b >= top.b && mid.b >= bottom.b && mid.b > THRESH) ? 1.0 : 0.0;
  flags.a = (mid.a >= right.a && mid.a >= left.a && mid.a >= top.a && mid.a >= bottom.a && mid.a > THRESH) ? 1.0 : 0.0;
  return flags * mid;
}


void main(void) {
  fragmentColor0 = process(inputLayer0);
#if NUM_LANES > 1
  fragmentColor1 = process(inputLayer1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = process(inputLayer2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = process(inputLayer3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = process(inputLayer4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = process(inputLayer5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = process(inputLayer6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = process(inputLayer7);
#endif
}
