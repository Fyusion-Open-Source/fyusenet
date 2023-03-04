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
#else
uniform sampler2D inputLayer0;
#endif

#ifdef HIGH_PRECISION
layout(location=0) out highp vec4 fragmentColor0;
#else
layout(location=0) out vec4 fragmentColor0;
#endif

in highp vec2 texCoord;
