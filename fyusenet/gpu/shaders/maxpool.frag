/* ----------------------------------------------------------------------------
 * Max-Pooling                             Copyright (c) 2016-2022 Fyusion Inc.
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

uniform ivec2 padding;

#include "shaders/activation.inc"

vec4 pool(in sampler2D inLayer,in ivec2 tc) {
#if POOL_SIZE == 2
  vec4 result = activate(texelFetch(inLayer,tc,0));
  result = max(result,activate(texelFetch(inLayer,tc+ivec2(1,0),0)));
  result = max(result,activate(texelFetch(inLayer,tc+ivec2(0,1),0)));
  result = max(result,activate(texelFetch(inLayer,tc+ivec2(1,1),0)));
  return result;
#else
#error NOT SUPPORTED YET
#endif
}

void main(void) {
  ivec2 tc = DOWNSAMPLE*(ivec2(gl_FragCoord.xy-vec2(0.5,0.5))-ivec2(padding.y,padding.y));
  tc+=ivec2(padding.x,padding.x);
  fragmentColor0 = pool(inputLayer0,tc);
#if NUM_LANES > 1
  fragmentColor1 = pool(inputLayer1,tc);
#endif
#if NUM_LANES > 2
  fragmentColor2 = pool(inputLayer2,tc);
#endif
#if NUM_LANES > 3
  fragmentColor3 = pool(inputLayer3,tc);
#endif
#if NUM_LANES > 4
  fragmentColor4 = pool(inputLayer4,tc);
#endif
#if NUM_LANES > 5
  fragmentColor5 = pool(inputLayer5,tc);
#endif
#if NUM_LANES > 6
  fragmentColor6 = pool(inputLayer6,tc);
#endif
#if NUM_LANES > 7
  fragmentColor7 = pool(inputLayer7,tc);
#endif
}
