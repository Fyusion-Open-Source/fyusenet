/* -------------------------------------------------------------------------------------------------
 * Masked SoftMax for Self-Attention Layer (Pass 1/2)                          (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D inputLayer0;
#else
uniform sampler2D inputLayer0;
#endif

layout(location=0) out vec4 fragmentColor0;

in highp vec2 tileCoord;            // x: token index (+1) for row ; y: y-coordinate in input texture
flat in highp int smXOffset;        // batch offset along the x-coordinate of the dot-prod texture for inner batches
uniform highp ivec2 inputParams;    // x: key length ,  y: query length

void main(void) {
    int col = smXOffset;
    int bmax = min(inputParams.x, min(int(tileCoord.x), col + INNER_BATCH_SIZE-1));
    if (bmax < col) discard;
    int row = int(tileCoord.y);
    vec4 accu = vec4(0.0);
    for (int b=col; b <= bmax; b++) {
        accu += clamp(exp(texelFetch(inputLayer0, ivec2(b, row), 0)), vec4(-FLT_MAX),vec4(FLT_MAX));
    }
    fragmentColor0 = accu;
}


