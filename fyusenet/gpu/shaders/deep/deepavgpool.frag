/* ----------------------------------------------------------------------------
 * Average-Pool Shader (Deep)               Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"

uniform highp float rescale;

uniform vec2 texStep;

#include "shaders/activation.inc"

#ifdef POOLSIZE
void main(void) {
#if POOLSIZE == 2
  vec4 up = activate(texture(inputLayer0,texCoord)) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,0)));
  vec4 dn = activate(textureOffset(inputLayer0,texCoord,ivec2(0,1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,1)));
  fragmentColor0 = 0.25*(up+dn);
#endif
#if POOLSIZE == 3
  vec4 up = activate(textureOffset(inputLayer0,texCoord,ivec2(-1,-1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(0,-1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,-1)));
  vec4 mid = activate(textureOffset(inputLayer0,texCoord,ivec2(-1,0))) + activate(textureOffset(inputLayer0,texCoord,ivec2(0,0))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,0)));
  vec4 dn = activate(textureOffset(inputLayer0,texCoord,ivec2(-1,1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(0,1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,1)));
  fragmentColor0 = (up+mid+dn)/9.0;
#endif
#if POOLSIZE == 4
  vec4 up1 = activate(texture(inputLayer0,texCoord)) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,0)));
  vec4 up2 = activate(textureOffset(inputLayer0,texCoord,ivec2(2,0))) + activate(textureOffset(inputLayer0,texCoord,ivec2(3,0)));
  vec4 dn1 = activate(textureOffset(inputLayer0,texCoord,ivec2(0,1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,1)));
  vec4 dn2 = activate(textureOffset(inputLayer0,texCoord,ivec2(2,1))) + activate(textureOffset(inputLayer0,texCoord,ivec2(3,1)));
  vec4 upA = up1 + up2;
  vec4 dnA = dn1 + dn2;
  up1 = activate(textureOffset(inputLayer0,texCoord,ivec2(0,2))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,2)));
  up2 = activate(textureOffset(inputLayer0,texCoord,ivec2(2,2))) + activate(textureOffset(inputLayer0,texCoord,ivec2(3,2)));
  dn1 = activate(textureOffset(inputLayer0,texCoord,ivec2(0,3))) + activate(textureOffset(inputLayer0,texCoord,ivec2(1,3)));
  dn2 = activate(textureOffset(inputLayer0,texCoord,ivec2(2,3))) + activate(textureOffset(inputLayer0,texCoord,ivec2(3,3)));
  vec4 upB = up1 + up2;
  vec4 dnB = dn1 + dn2;
  vec4 up = upA + upB;
  vec4 dn = dnA + dnB;
  fragmentColor0 = (up+dn)/16.0;
#endif
}

#else

void main(void) {
  // NOTE (mw) not nice, find something better in the future
  vec4 m = vec4(0.0);
  for (int y=0; y < POOLSIZE_Y; y++) {
    for (int x=0; x < POOLSIZE_X; x++) {
      m += activate(texture(inputLayer0, texCoord+vec2(x,y)*texStep));
    }
  }
  fragmentColor0 = m / float(POOLSIZE_X*POOLSIZE_Y);
}

#endif

