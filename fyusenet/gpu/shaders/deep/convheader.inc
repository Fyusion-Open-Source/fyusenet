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
layout(binding=1) uniform sampler2D residualLayer0;
layout(binding=BIAS_UNIT) uniform sampler2D biasTexture;
#else
uniform sampler2D inputLayer0;
uniform sampler2D residualLayer0;
uniform sampler2D biasTexture;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec4 texCoord;

#if defined(USE_RESIDUAL)
in highp vec2 resCoord;
#endif
