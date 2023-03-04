/* ----------------------------------------------------------------------------
 * Generic Kernel Layer                     Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision mediump float;
precision lowp int;
precision mediump sampler2D;

#define MAX_KERNEL_SIZE 15

#if KERNEL_SIZE > MAX_KERNEL_SIZE
#error maximum kernel size exceeded
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

uniform vec4 kernelCoeffs[KERNEL_SIZE*KERNEL_SIZE];

#ifndef NO_ACT
#error activations are not supported yet
#endif

#ifdef POST_BATCHNORM
#error batchnorm is not supported yet
#endif

vec4 convolve(in sampler2D sampler) {
  vec4 result = vec4(0);
#if KERNEL_SIZE==3
  result += kernelCoeffs[0]*textureOffset(sampler,texCoord,ivec2(-1,-1));
  result += kernelCoeffs[1]*textureOffset(sampler,texCoord,ivec2( 0,-1));
  result += kernelCoeffs[2]*textureOffset(sampler,texCoord,ivec2( 1,-1));
  result += kernelCoeffs[3]*textureOffset(sampler,texCoord,ivec2(-1, 0));
  result += kernelCoeffs[4]*textureOffset(sampler,texCoord,ivec2( 0, 0));
  result += kernelCoeffs[5]*textureOffset(sampler,texCoord,ivec2( 1, 0));
  result += kernelCoeffs[6]*textureOffset(sampler,texCoord,ivec2(-1, 1));
  result += kernelCoeffs[7]*textureOffset(sampler,texCoord,ivec2( 0, 1));
  result += kernelCoeffs[8]*textureOffset(sampler,texCoord,ivec2( 1, 1));
#else
  // larger kernels
  int kidx=0;
  for (int y=-(KERNEL_SIZE-1)/2 ; y<(KERNEL_SIZE-1)/2 ; y++) {
    for (int x=-(KERNEL_SIZE-1)/2 ; x<(KERNEL_SIZE-1)/2 ; x++) {
      result += kernelCoeffs[kidx++]*textureOffset(sampler,texCoord,ivec2(x,y));
    }
  }
#endif
  return result;
}

void main(void) {
  fragmentColor0 = convolve(inputLayer0);
#if NUM_LANES > 1
  fragmentColor1 = convolve(inputLayer1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = convolve(inputLayer2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = convolve(inputLayer3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = convolve(inputLayer4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = convolve(inputLayer5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = convolve(inputLayer6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = convolve(inputLayer7);
#endif
}
