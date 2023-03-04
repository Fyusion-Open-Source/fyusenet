/* ----------------------------------------------------------------------------
 * Maxpooling (Deep Tensor Format)        Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"
#include "shaders/activation.inc"

uniform vec2 texStep;

#ifdef POOLSIZE
void main(void) {
#if POOLSIZE == 2
  vec4 up = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,-PADDING))));
  vec4 dn = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,1-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,1-PADDING))));
  fragmentColor0 = max(up,dn);
#endif
#if POOLSIZE == 3
  vec4 up = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,-PADDING))));
  vec4 mid = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,1-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,1-PADDING))));
  vec4 dn = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,2-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,2-PADDING))));
  up = max(up,textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,-PADDING)));
  mid = max(mid,textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,1-PADDING)));
  dn = max(dn,textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,2-PADDING)));
  up = max(up,mid);
  fragmentColor0 = max(up,dn);
#endif
#if POOLSIZE == 4
  vec4 up1 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,-PADDING))));
  vec4 up2 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(3-PADDING,-PADDING))));
  vec4 dn1 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,1-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,1-PADDING))));
  vec4 dn2 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,1-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(3-PADDING,1-PADDING))));
  vec4 upA = max(up1,up2);
  vec4 dnA = max(dn1,dn2);
  up1 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,2-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,2-PADDING))));
  up2 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,2-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(3-PADDING,2-PADDING))));
  dn1 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(-PADDING,3-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(1-PADDING,3-PADDING))));
  dn2 = max(activate(textureOffset(inputLayer0,texCoord,ivec2(2-PADDING,3-PADDING))), activate(textureOffset(inputLayer0,texCoord,ivec2(3-PADDING,3-PADDING))));
  vec4 upB = max(up1,up2);
  vec4 dnB = max(dn1,dn2);
  vec4 up = max(upA,upB);
  vec4 dn = max(dnA,dnB);
  fragmentColor0 = max(up,dn);
#endif
}

#else
// pooling for general (larger) sizes
void main(void) {
  // NOTE (mw) not nice, find something better in the future
  vec4 m = activate(texture(inputLayer0, texCoord));
  for (int y=-PADDING;y<POOLSIZE_Y-PADDING;y++) {
    for (int x=-PADDING;x<POOLSIZE_X-PADDING;x++) {
      m = max(m,activate(texture(inputLayer0, texCoord+vec2(x,y)*texStep)));
    }
  }
  fragmentColor0 = m;
}

#endif

