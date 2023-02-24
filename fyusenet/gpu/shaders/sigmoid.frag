/* ----------------------------------------------------------------------------
 * Sigmoid Layer (Shallow)                  Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/funcpreamble.inc"
#include "shaders/activation.inc"

vec4 process(in sampler2D sampler) {
    vec4 data = activate(texture(sampler, texCoord));
    vec4 outpix = 1.0/(1.0+exp(-data));
    return outpix;
}

void main(void) {
    fragmentColor0 = process(inputLayer0);
#if NUM_LANES > 1
    fragmentColor1 = process(inputLayer1);
#endif
#if NUM_LANES > 2
    fragmentColor2 = process(inputLayer2);
#endif
#if NUM_LANES > 3
    fragmentColor3 = process(inputLayer3);
#endif
#if NUM_LANES > 4
    fragmentColor4 = process(inputLayer4);
#endif
#if NUM_LANES > 5
    fragmentColor5 = process(inputLayer5);
#endif
#if NUM_LANES > 6
    fragmentColor6 = process(inputLayer6);
#endif
#if NUM_LANES > 7
    fragmentColor7 = process(inputLayer7);
#endif
}
