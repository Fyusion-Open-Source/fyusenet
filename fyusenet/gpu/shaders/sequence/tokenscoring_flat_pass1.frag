/* -------------------------------------------------------------------------------------------------
 * Pass 1 Flattening for Token Scoring                                         (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

/* This shader aggregates token scoring values from the full "scoring sheet" (a texture that
   stores the inner-product of the output vector with each of the embedding vectors in the
   vocabulary) using very basic statistics. Instead of contracting to a single output
   right away, this step will contract to another (smaller) texture in order to distribute the
   work a bit better across the SMs. For each "contraction range", the output will consist of:
    - maximum value
    - minimum value
    - average value
    - standard deviation
    - number of values greater or equal than a 95% mix of maximum and mean
    - number of values greater or equal than a 90% mix of maximum and mean
    - number of values greater or equal than a 75% mix of maximum and mean
    - number of values greater or equal than a 50% mix of maximum and mean
*/

#ifndef HIGH_PRECISION
precision mediump sampler2D;
#else
precision highp sampler2D;
#endif

precision highp float;
precision highp int;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D projection;
#else
uniform sampler2D projection;
#endif

layout(location=0) out vec4 fragmentColor0;
layout(location=1) out vec4 fragmentColor1;

in highp vec2 texturePos;

uniform ivec2 contractionRange;
vec4 rangeBuffer[BUFFER_SIZE];

void main(void) {
    highp vec2 minimax = vec2(-FLT_MAX, FLT_MAX);
    highp float mean = 0.0;
    int num = contractionRange.x * contractionRange.y;
    for (int y=0, bidx=0; y < contractionRange.y; y++) {
        for (int x=0; x < contractionRange.x; x++, bidx++) {
            vec4 val = texelFetch(projection, ivec2(texturePos) + ivec2(x,y), 0);
            val.rg = (val.r < val.g) ? val.rg : val.gr;
            val.rb = (val.r < val.b) ? val.rb : val.br;
            val.ra = (val.r < val.a) ? val.ra : val.ar;
            val.gb = (val.g < val.b) ? val.gb : val.bg;
            val.ga = (val.g < val.a) ? val.ga : val.ag;
            val.ba = (val.b < val.a) ? val.ba : val.ab;
            rangeBuffer[bidx] = val;
            minimax = vec2(max(minimax.r, val.a), min(minimax.g, val.r));
            mean += dot(val, vec4(1));
        }
    }
    mean /= float(num*4);
    float variance = 0.0;
    vec4 meantests = vec4(minimax.r*0.95 + mean*0.05,
                          minimax.r*0.9 + mean*0.1,
                          minimax.r*0.75 + mean*0.25,
                          minimax.r*0.5 + mean*0.5);
    float numg95 = 0.0;
    float numg90 = 0.0;
    float numg75 = 0.0;
    float numg50 = 0.0;
    for (int i=0; i < num; i++) {
        vec4 val = rangeBuffer[i];
        vec4 off = val - vec4(mean);
        variance += dot(off, off);
        numg95 += dot(vec4(greaterThanEqual(val, meantests.rrrr)), vec4(1.0));
        numg90 += dot(vec4(greaterThanEqual(val, meantests.gggg)), vec4(1.0));
        numg75 += dot(vec4(greaterThanEqual(val, meantests.bbbb)), vec4(1.0));
        numg50 += dot(vec4(greaterThanEqual(val, meantests.aaaa)), vec4(1.0));
    }
    variance /= float(4*num-1);

    fragmentColor0 = vec4(minimax.g, minimax.r, mean, sqrt(variance));    // min, max, mean, stddev
    fragmentColor1 = vec4(numg95, numg90, numg75, numg50);
}
