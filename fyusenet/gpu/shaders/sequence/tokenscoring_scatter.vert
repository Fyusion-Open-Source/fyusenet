/* -------------------------------------------------------------------------------------------------
 * Token Scoring Scatter / Sorting Step                                        (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=0) uniform sampler2D projection;
layout(binding=1) uniform sampler2D stats;
#else
uniform sampler2D projection;
uniform sampler2D stats;
#endif

in highp uint attributes0;

flat out highp uint tokenIndex;
flat out highp vec4 match;

uniform highp ivec2 projSize;
uniform highp vec2 scatterShift;

vec2 narrowRange(in vec4 params1, in vec4 params2) {
    float low = params1.x;
    float entries = 0.0;
    if (params2.y > 0.0) {
        low = (params1.y * 0.95 + params2.x * 0.05);
        entries = params2.y;
    } else if (params2.z > 0.0) {
        low = (params1.y * 0.9 + params2.x * 0.1);
        entries = params2.z;
    } else if (params2.w > 0.0) {
        low = (params1.y * 0.75 + params2.x * 0.25);
        entries = params2.w;
    }
    // TODO (mw) some more guesstimating here if num entries is too large
    if (entries < 1.0) low = params2.x;
    return vec2(low, params1.y);
}

void main(void) {
    int offset = int(attributes0 / 4u);
    int sub = int(attributes0 % 4u);
    int px = offset % projSize.x;
    int py = offset / projSize.x;
    vec4 pdata = texelFetch(projection, ivec2(px, py), 0);
    vec4 params1 = texelFetch(stats, ivec2(0, 0), 0);       // min, max, mean, stddev
    vec4 params2 = texelFetch(stats, ivec2(1, 0), 0);       // maxmin, 95pcount, 90pcount, 75pcount
    vec2 range;
    float row = 0.0;
    vec2 narrow = narrowRange(params1, params2);
    vec2 wide = vec2(params2.x, narrow.x);
    if (gl_InstanceID == 0) {
        row = -0.75;  // shift included
        range = narrow;
    } else range = wide;
    float position = pdata[sub] - range.x;
    float span = max(1e-7, range.y - range.x);      // FIXME (mw) 1e-7 is a really narrow interval, should not happen
    float t = -1.0;
    if ((position < 0.0) || ((gl_InstanceID == 0) && (position > span)) ||
        (gl_InstanceID > 0) && (position >= span)) {
        gl_Position = vec4(0.0);
        t = 2.0;
        row = 0.75;  // shift included
    } else {
        t = 1.0 - min(1.0, position/span);
        gl_Position = vec4(t*2.0-1.0 + scatterShift.x, row, t, 1.0);
    }
    match.x = pdata[sub];
    match.y = float(px);
    match.z = float(py);
    match.w = float(t);
    tokenIndex = attributes0 + 1u;
    gl_PointSize = 1.0;
}

