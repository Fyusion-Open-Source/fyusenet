
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

