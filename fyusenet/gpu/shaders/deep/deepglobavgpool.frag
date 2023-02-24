/* ----------------------------------------------------------------------------
 * Global AvgPool (Deep Tensor Format)     Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/deep/fragpreamble.inc"
#include "shaders/activation.inc"

uniform highp ivec2 imdim;
uniform highp vec2 texStep;

// FIXME (mw) this will be very bad on larger image dimensions, find a better solution
// for those cases
void main(void) {
  highp vec4 accu = vec4(0);
  highp vec2 tc = texCoord + texStep/2.0;
  for (int y=0; y < imdim.y; y++) {
    for (int x=0; x < imdim.x; x++) {
      accu += activate(texture(inputLayer0, tc+vec2(x*texStep.x, y*texStep.y)));
    }
  }
  fragmentColor0 = accu / float(imdim.x * imdim.y);
}
