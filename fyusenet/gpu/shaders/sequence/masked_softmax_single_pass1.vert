/* -------------------------------------------------------------------------------------------------
 * Masked SoftMax for Self-Attention Layer (Pass 1/2), single token            (c) Martin Wawro 2023
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ---------------------------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

in highp uint attributes0;          // column y-coordinate

out highp float tileRow;            // row number in the input texture
flat out highp int smXOffset;       // batch offset along the x-coordinate for inner batches

uniform highp vec2 viewport;        // x,y: viewport size

void main(void) {
    tileRow = float(attributes0) * viewport.y;
    float fuzz = (attributes0 > 0u) ? 0.25 : 0.0;
    gl_Position = vec4(0.0, 2.0 * (tileRow + fuzz)/viewport.y - 1.0,  0.0, 1.0);     // see "diamond-exit" rule in GL spec section 3.4.1
    smXOffset = gl_InstanceID * INNER_BATCH_SIZE;
}

