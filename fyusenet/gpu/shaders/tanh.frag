/* ----------------------------------------------------------------------------
 * tanh Activation (Common)                Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/funcpreamble.inc"
#include "shaders/activation.inc"

vec4 process(in sampler2D sampler,in const int bidx) {
    vec4 data = activate(texture(sampler, texCoord));
    data = 2.0*(exp(2.0*data)/(1.0+exp(2.0*data)))-1.0;
    return data;
}

void main(void) {
    fragmentColor0 = process(inputLayer0, 0);
#if NUM_LANES > 1
    fragmentColor1 = process(inputLayer1, 2);
#endif
#if NUM_LANES > 2
    fragmentColor2 = process(inputLayer2, 4);
#endif
#if NUM_LANES > 3
    fragmentColor3 = process(inputLayer3, 6);
#endif
#if NUM_LANES > 4
    fragmentColor4 = process(inputLayer4, 8);
#endif
#if NUM_LANES > 5
    fragmentColor5 = process(inputLayer5, 10);
#endif
#if NUM_LANES > 6
    fragmentColor6 = process(inputLayer6, 12);
#endif
#if NUM_LANES > 7
    fragmentColor7 = process(inputLayer7, 14);
#endif
}
