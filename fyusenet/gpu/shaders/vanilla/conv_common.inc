#ifdef MALI_WORKAROUND
precision highp float;
precision highp int;
precision highp sampler2D;
#else
#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#else
precision highp float;
precision highp int;
precision highp sampler2D;
#endif
#endif

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer;
#else
uniform mediump sampler2D inputLayer;
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

#ifdef USE_RESIDUAL
#ifdef BINDING_SUPPORT
layout(binding=1) uniform sampler2D resLayer0;
layout(binding=2) uniform sampler2D resLayer1;
layout(binding=3) uniform sampler2D resLayer2;
layout(binding=4) uniform sampler2D resLayer3;
layout(binding=5) uniform sampler2D resLayer4;
layout(binding=6) uniform sampler2D resLayer5;
layout(binding=7) uniform sampler2D resLayer6;
layout(binding=8) uniform sampler2D resLayer7;
#else
uniform sampler2D resLayer0;
uniform sampler2D resLayer1;
uniform sampler2D resLayer2;
uniform sampler2D resLayer3;
uniform sampler2D resLayer4;
uniform sampler2D resLayer5;
uniform sampler2D resLayer6;
uniform sampler2D resLayer7;
#endif
#endif

in highp vec2 texCoord;

#ifdef USE_RESIDUAL
in highp vec2 resCoord;
uniform int addResidual;
#endif

uniform mat4 coeffs[CONVSIZE*NUM_LANES];

#ifdef USE_BIAS
uniform vec4 bias[NUM_LANES];
#endif

#ifdef POST_BATCHNORM
uniform vec4 batchnorm[NUM_LANES];
#endif

