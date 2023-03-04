/* ----------------------------------------------------------------------------
 * 1x1 Shallow Convolution (Vanilla)       Copyright (c) 2016-2022 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

#include "shaders/vanilla/conv_common.inc"
#include "shaders/activation.inc"
#include "shaders/vanilla/residual.inc"

void main(void) {  
  vec4 pix = activate(texture(inputLayer,texCoord));
#ifdef POST_BATCHNORM
  fragmentColor0 = batchnorm[0]*(coeffs[0]*pix);
#else
  fragmentColor0 = coeffs[ 0]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor0 += bias[0];
#endif

#if NUM_LANES > 1
#ifdef POST_BATCHNORM
  fragmentColor1 = batchnorm[1]*(coeffs[1]*pix);
#else
  fragmentColor1 = coeffs[ 1]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor1 += bias[1];
#endif
#endif // LANES

#if NUM_LANES > 2
#ifdef POST_BATCHNORM
  fragmentColor2 = batchnorm[2]*(coeffs[2]*pix);
#else
  fragmentColor2 = coeffs[ 2]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor2 += bias[2];
#endif
#endif // LANES


#if NUM_LANES > 3
#ifdef POST_BATCHNORM
  fragmentColor3 = batchnorm[3]*(coeffs[3]*pix);
#else
  fragmentColor3 = coeffs[ 3]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor3 += bias[3];
#endif
#endif // LANES

#if NUM_LANES > 4
#ifdef POST_BATCHNORM
  fragmentColor4 = batchnorm[4]*(coeffs[4]*pix);
#else
  fragmentColor4 = coeffs[ 4]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor4 += bias[4];
#endif
#endif // LANES

#if NUM_LANES > 5
#ifdef POST_BATCHNORM
  fragmentColor5 = batchnorm[5]*(coeffs[5]*pix);
#else
  fragmentColor5 = coeffs[ 5]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor5 += bias[5];
#endif
#endif // LANES

#if NUM_LANES > 6
#ifdef POST_BATCHNORM
  fragmentColor6 = batchnorm[6]*(coeffs[6]*pix);
#else
  fragmentColor6 = coeffs[ 6]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor6 += bias[6];
#endif
#endif // LANES

#if NUM_LANES > 7
#ifdef POST_BATCHNORM
  fragmentColor7 = batchnorm[7]*(coeffs[7]*pix);
#else
  fragmentColor7 = coeffs[ 7]*pix;
#endif
#ifdef USE_BIAS
  fragmentColor7 += bias[7];
#endif
#endif // LANES

#ifdef USE_RESIDUAL
  if (addResidual > 0) handleResidual();
#endif  // USE_RESIDUAL
}
