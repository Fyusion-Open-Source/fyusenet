/* -------------------------------------------------------------------------------------------------
 * Embedding Layer for Sequences                                              (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

#ifndef HIGH_PRECISION
precision mediump float;
precision mediump int;
precision mediump sampler2D;
#else
precision highp float;
precision highp int;
precision highp sampler2D;
#endif

#ifdef BINDING_SUPPORT
layout(binding=1) uniform sampler2D vocabulary0;
#if VOCAB_SIZE > 1
layout(binding=2) uniform sampler2D vocabulary1;
#endif
#if VOCAB_SIZE > 2
layout(binding=3) uniform sampler2D vocabulary2;
#endif
#if VOCAB_SIZE > 3
layout(binding=4) uniform sampler2D vocabulary3;
#endif
#if VOCAB_SIZE > 4
layout(binding=5) uniform sampler2D vocabulary4;
#endif
#if VOCAB_SIZE > 5
layout(binding=6) uniform sampler2D vocabulary5;
#endif
#if VOCAB_SIZE > 6
layout(binding=7) uniform sampler2D vocabulary6;
#endif
#if VOCAB_SIZE > 7
layout(binding=8) uniform sampler2D vocabulary7;
#endif
#else
uniform sampler2D vocabulary0;
#if VOCAB_SIZE > 1
uniform sampler2D vocabulary1;
#endif
#if VOCAB_SIZE > 2
uniform sampler2D vocabulary2;
#endif
#if VOCAB_SIZE > 3
uniform sampler2D vocabulary3;
#endif
#if VOCAB_SIZE > 4
uniform sampler2D vocabulary4;
#endif
#if VOCAB_SIZE > 5
uniform sampler2D vocabulary5;
#endif
#if VOCAB_SIZE > 6
uniform sampler2D vocabulary6;
#endif
#if VOCAB_SIZE > 7
uniform sampler2D vocabulary7;
#endif
#endif


layout(location=0) out vec4 fragmentColor0;

in highp float textureX;
flat in highp ivec2 textureYZ;     // x: row in selected vocabulary texture, y: index for vocabulary texture

void main(void) {
    highp ivec2 pos = ivec2(textureX, textureYZ.x);
#if VOCAB_SIZE == 1
    fragmentColor0 = texelFetch(vocabulary0, pos, 0);
#else
    switch (textureYZ.y) {
        case 0:
            fragmentColor0 = texelFetch(vocabulary0, pos, 0);
            break;
#if VOCAB_SIZE > 1
        case 1:
            fragmentColor0 = texelFetch(vocabulary1, pos, 0);
            break;
#endif
#if VOCAB_SIZE > 2
        case 2:
            fragmentColor0 = texelFetch(vocabulary2, pos, 0);
            break;
#endif
#if VOCAB_SIZE > 3
        case 3:
            fragmentColor0 = texelFetch(vocabulary3, pos, 0);
            break;
#endif
#if VOCAB_SIZE > 4
        case 4:
            fragmentColor0 = texelFetch(vocabulary4, pos, 0);
            break;
#endif
#if VOCAB_SIZE > 5
        case 5:
            fragmentColor0 = texelFetch(vocabulary5, pos, 0);
            break;
#endif
#if VOCAB_SIZE > 6
        case 6:
            fragmentColor0 = texelFetch(vocabulary6, pos, 0);
            break;
        #endif
#if VOCAB_SIZE > 7
        case 7:
            fragmentColor0 = texelFetch(vocabulary7, pos, 0);
            break;
#endif
        default:
            discard;
    }
#endif
}
