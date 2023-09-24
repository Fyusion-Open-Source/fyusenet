/* -------------------------------------------------------------------------------------------------
 * Rotary Encoding as Positional Encoding for Sequences                        (c) Martin Wawro 2023
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

in highp vec2 inputPos;

uniform highp float thetaBase;     // base value for computing theta as defined in the RoPE paper
uniform highp ivec2 headDim;       // x: pixel head_dim, y: actual head_dim
uniform highp int tokenIdx;        // token index

void main(void) {
    int head = int(inputPos.x) / headDim.x;
    int headoffset = int(inputPos.x) % headDim.x;
    int y = int(inputPos.y);
    int headbase = head * headDim.x;
    ivec2 unrotated = ivec2(headbase + headoffset, y);
    ivec2 rotated = ivec2(headbase + ((headoffset + headDim.x/2) % headDim.x), y);
    vec4 data = texelFetch(inputLayer0, unrotated, 0);
    vec4 datar = texelFetch(inputLayer0, rotated, 0);
    float sg = float(sign(headbase + headoffset - rotated.x));
    vec4 p = vec4((ivec4(2 * headoffset * 4) + ivec4(0,2,4,6)) % headDim.y);
    vec4 freqs = float(y + tokenIdx) * pow(vec4(thetaBase), -p / float(headDim.y));
    vec4 cfreqs = cos(freqs);
    vec4 sfreqs = sin(freqs);
    fragmentColor0 = data * cfreqs + sg * datar * sfreqs;
}
