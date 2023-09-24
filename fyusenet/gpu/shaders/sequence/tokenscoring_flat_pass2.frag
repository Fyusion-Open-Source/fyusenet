/* -------------------------------------------------------------------------------------------------
 * Token Selection/Sampling Flattening Pass 2/2                                (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D pass1DataA;
layout(binding=1) uniform sampler2D pass1DataB;
#else
uniform sampler2D pass1DataA;
uniform sampler2D pass1DataB;
#endif

layout(location=0) out vec4 fragmentColor0;

uniform ivec2 contractionRange;

float allmax, allmin, allmean, allstd;
vec4 maxPatchA, maxPatchB;

void computeBasics() {
    allmax = -FLT_MAX;
    allmin = FLT_MAX;
    allmean = 0.0;
    allstd = 0.0;
    for (int y=0; y < contractionRange.y; y++) {
        for (int x=0; x < contractionRange.x; x++) {
            vec4 valA = texelFetch(pass1DataA, ivec2(x, y), 0);
            if (valA.y > allmax) {
                vec4 valB = texelFetch(pass1DataB, ivec2(x, y), 0);
                maxPatchA = valA;
                maxPatchB = valB;
                allmax = valA.y;
            }
            allmin = min(valA.x, allmin);
            allmean += valA.z;
            allstd = max(allstd, valA.a);// well, that is not correct but it works
        }
    }
    allmean /= float(contractionRange.x * contractionRange.y);
}

void main() {
    computeBasics();
    if (gl_FragCoord.x < 1.0) {
        fragmentColor0 = vec4(allmin, allmax, allmean, allstd);
    } else {
        fragmentColor0 = vec4(maxPatchA.x, maxPatchB.x, maxPatchB.y, maxPatchB.z);
    }
}
