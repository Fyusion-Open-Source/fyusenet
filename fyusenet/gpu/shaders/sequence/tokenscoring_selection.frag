/* -------------------------------------------------------------------------------------------------
 * Token-Scoring Selection                                                     (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform usampler2D tokenData;
#else
uniform usampler2D tokenData;
#endif

layout(location=0) out uint fragmentColor0;

#ifdef TOP_K
uniform int topk;
uniform highp int seed;
#endif

uint greedyFetch() {
    for (int i=0; i < SCATTER_WIDTH; i++) {
        uint tokenIndex = texelFetch(tokenData, ivec2(i, 0), 0).r;
        if (tokenIndex > 0u) return tokenIndex-1u;
    }
    return 0u;   // FIXME (mw) we should return an EOS token here to prevent problems from piling up
}

void main(void) {
#ifdef GREEDY
    fragmentColor0 = greedyFetch();
#endif
#ifdef TOP_K
    // TODO (mw) use the probability to select the token, not just a uniform random
    int k = min(topk-1, int(floor(rand(seed) * float(topk))));
    int cnt = 0;
    for (int i=0; i < SCATTER_WIDTH; i++) {
        uint tokenIndex = texelFetch(tokenData, ivec2(i, 0), 0).r;
        if (tokenIndex > 0) {
            if (k == cnt) {
                fragmentColor0 = tokenIndex-1;
                cnt = -1;
                break;
            }
            cnt++;
        }
    }
    if (cnt != -1) {
        for (int i=0; i < SCATTER_WIDTH; i++) {
            uint tokenIndex = texelFetch(tokenData, ivec2(i, 1), 0).r;
            if (tokenIndex > 0) {
                if (k == cnt) {
                    fragmentColor0 = tokenIndex-1;
                    cnt = -1;
                    break;
                }
                cnt++;
            }
        }
    }
    if (cnt == -1) {
        // fall back to greedy
        fragmentColor0 = greedyFetch();
    }
#endif
}
