/* -------------------------------------------------------------------------------------------------
 * Masked SoftMax for Self-Attention Layer (Pass 1/2), batched                 (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

in highp uint attributes0;          // column y-coordinate

out highp vec2 tileCoord;           // x: token index (+1) for row ; y: y-coordinate in input dot-prod texture
flat out highp int smXOffset;       // batch offset along the x-coordinate of the dot-prod texture for inner batches

uniform highp vec2 viewport;        // x,y: viewport size (x currently fixed to 1)
uniform highp ivec2 inputParams;    // x: key length,  y: query length
uniform highp int baseTokenIdx;     // index of the first query token (corresponding to generic y == 0), used for masking

void main(void) {
    float y = float(inputParams.y * int(attributes0 & 0xFFFFu));
    int endpoint = int((attributes0 >> 16) & 0x1u);
    float fuzz = 0.25;
    float gly = 2.0 * ((y+fuzz) / viewport.y) - 1.0;  // see "diamond-exit" rule in GL spec section 3.4.1
    gl_Position = vec4(0.0, gly,  0.0, 1.0);
    smXOffset = gl_InstanceID * INNER_BATCH_SIZE;
    int token = endpoint * inputParams.y;
    tileCoord.xy = vec2(baseTokenIdx+token, y);
}

