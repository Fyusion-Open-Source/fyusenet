vec4 multiply(in vec4 inpix,in int offset,const in int bnscale) {
#ifdef POST_BATCHNORM
  return coeffs[offset]*inpix*batchnorm[bnscale];
#else
  return coeffs[offset]*inpix;
#endif
}


void procAndAdd(in vec4 inpix,in int offset) {
  vec4 pix = activate(inpix);
  fragmentColor0 += multiply(pix,offset,0);
#if NUM_LANES > 1
  fragmentColor1 += multiply(pix,offset+CONVSIZE,1);
#endif
#if NUM_LANES > 2
  fragmentColor2 += multiply(pix,offset+2*CONVSIZE,2);
#endif
#if NUM_LANES > 3
  fragmentColor3 += multiply(pix,offset+3*CONVSIZE,3);
#endif
#if NUM_LANES > 4
  fragmentColor4 += multiply(pix,offset+4*CONVSIZE,4);
#endif
#if NUM_LANES > 5
  fragmentColor5 += multiply(pix,offset+5*CONVSIZE,5);
#endif
#if NUM_LANES > 6
  fragmentColor6 += multiply(pix,offset+6*CONVSIZE,6);
#endif
#if NUM_LANES > 7
  fragmentColor7 += multiply(pix,offset+7*CONVSIZE,7);
#endif
}


void procAndSet(in vec4 inpix,in int offset) {
  vec4 pix = activate(inpix);
#ifdef USE_BIAS
  fragmentColor0 = bias[0]+multiply(pix,offset,0);
#if NUM_LANES > 1
  fragmentColor1 = bias[1]+multiply(pix,offset+CONVSIZE,1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = bias[2]+multiply(pix,offset+2*CONVSIZE,2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = bias[3]+multiply(pix,offset+3*CONVSIZE,3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = bias[4]+multiply(pix,offset+4*CONVSIZE,4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = bias[5]+multiply(pix,offset+5*CONVSIZE,5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = bias[6]+multiply(pix,offset+6*CONVSIZE,6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = bias[7]+multiply(pix,offset+7*CONVSIZE,7);
#endif
#else
  fragmentColor0 = multiply(pix,offset,0);
#if NUM_LANES > 1
  fragmentColor1 = multiply(pix,offset+CONVSIZE,1);
#endif
#if NUM_LANES > 2
  fragmentColor2 = multiply(pix,offset+2*CONVSIZE,2);
#endif
#if NUM_LANES > 3
  fragmentColor3 = multiply(pix,offset+3*CONVSIZE,3);
#endif
#if NUM_LANES > 4
  fragmentColor4 = multiply(pix,offset+4*CONVSIZE,4);
#endif
#if NUM_LANES > 5
  fragmentColor5 = multiply(pix,offset+5*CONVSIZE,5);
#endif
#if NUM_LANES > 6
  fragmentColor6 = multiply(pix,offset+6*CONVSIZE,6);
#endif
#if NUM_LANES > 7
  fragmentColor7 = multiply(pix,offset+7*CONVSIZE,7);
#endif
#endif
}

