uniform highp float texStep;

vec4 multiply(in vec4 inpix, in int offset, const in int bnscale) {
#ifdef POST_BATCHNORM
  return coeffs[offset]*inpix*batchnorm[bnscale];
#else
  return coeffs[offset]*inpix;
#endif
}

void process(in vec4 inpix) {
  inpix = activate(inpix);
#ifdef USE_BIAS
  fragmentColor0 = bias[0] + multiply(inpix,0,0);
#else
  fragmentColor0 = multiply(inpix, 0, 0);
#endif
#if NUM_LANES > 1
#ifdef USE_BIAS
  fragmentColor1 = bias[1] + multiply(inpix, CONVSIZE, 1);
#else
  fragmentColor1 = multiply(inpix, CONVSIZE, 1);
#endif
#endif
#if NUM_LANES > 2
#ifdef USE_BIAS
  fragmentColor2 = bias[2] + multiply(inpix, 2*CONVSIZE, 2);
#else
  fragmentColor2 = multiply(inpix, 2*CONVSIZE, 2);
#endif
#endif
#if NUM_LANES > 3
#ifdef USE_BIAS
  fragmentColor3 = bias[3] + multiply(inpix, 3*CONVSIZE, 3);
#else
  fragmentColor3 = multiply(inpix, 3*CONVSIZE, 3);
#endif
#endif
#if NUM_LANES > 4
#ifdef USE_BIAS
  fragmentColor4 = bias[4] + multipy(inpix, 4*CONVSIZE, 4);
#else
  fragmentColor4 = multiply(inpix, 4*CONVSIZE, 4);
#endif
#endif
#if NUM_LANES > 5
#ifdef USE_BIAS
  fragmentColor5 = bias[5] + multiply(inpix, 5*CONVSIZE, 5);
#else
  fragmentColor5 = multiply(inpix, 5*CONVSIZE, 5);
#endif
#endif
#if NUM_LANES > 6
#ifdef USE_BIAS
  fragmentColor6 = bias[6] + multiply(inpix, 6*CONVSIZE, 6);
#else
  fragmentColor6 = multiply(inpix, 6*CONVSIZE, 6);
#endif
#endif
#if NUM_LANES > 7
#ifdef USE_BIAS
  fragmentColor7 = bias[7] + multiply(inpix, 7*CONVSIZE, 7);
#else
  fragmentColor7 = multiply(inpix, 7*CONVSIZE, 7);
#endif
#endif
}

void processAndAdd(in vec4 inpix,in int offset) {
  fragmentColor0 += multiply(inpix,offset, 0);
#if NUM_LANES > 1
  fragmentColor1 += multiply(inpix,offset+CONVSIZE, 1);
#endif
#if NUM_LANES > 2
  fragmentColor2 += multiply(inpix,offset+2*CONVSIZE, 2);
#endif
#if NUM_LANES > 3
  fragmentColor3 += multiply(inpix,offset+3*CONVSIZE, 3);
#endif
#if NUM_LANES > 4
  fragmentColor4 += multiply(inpix,offset+4*CONVSIZE, 4);
#endif
#if NUM_LANES > 5
  fragmentColor5 += multiply(inpix,offset+5*CONVSIZE, 5);
#endif
#if NUM_LANES > 6
  fragmentColor6 += multiply(inpix,offset+6*CONVSIZE, 6);
#endif
#if NUM_LANES > 7
  fragmentColor7 += multiply(inpix,offset+7*CONVSIZE, 7);
#endif
}
