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

in highp float tileRow;           // row number in the texture
flat in highp int smXOffset;      // batch offset along the x-coordinate for inner batches

uniform highp int keyLength;
uniform highp int tokenIdx;       // index of the query token

void main(void) {
    int col = smXOffset;
    int bmax = min(keyLength, min(tokenIdx, col + INNER_BATCH_SIZE -1));
    if (bmax < col) discard;
    int row = int(tileRow);
    vec4 accu = vec4(0.0);
    for (int b=col; b <= bmax; b++) {
        accu += clamp(exp(texelFetch(inputLayer0, ivec2(b, row), 0)), vec4(-FLT_MAX),vec4(FLT_MAX));
    }
    fragmentColor0 = accu;
}


