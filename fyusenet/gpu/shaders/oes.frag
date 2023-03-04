/* ----------------------------------------------------------------------------
 * OES Conversion (Common / EGL / Android) Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#ifdef GL_OES_EGL_image_external
#ifndef GL_OES_EGL_image_external_essl3
#extension GL_OES_EGL_image_external : enable
#endif
#endif
#ifdef GL_OES_EGL_image_external_essl3
#extension GL_OES_EGL_image_external : disable
#extension GL_OES_EGL_image_external_essl3 : enable
#endif


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
layout(binding=0) uniform samplerExternalOES inputLayer;
#else
uniform samplerExternalOES inputLayer;
#endif

in highp vec2 texCoord;

layout(location=0) out vec4 fragmentColor;

void main(void) {
  fragmentColor = texture(inputLayer,texCoord);
}
