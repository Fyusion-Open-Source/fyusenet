#ifdef USE_RESIDUAL
vec4 resFetch(in sampler2D sampler,in vec2 tc) {
#ifdef RELU_ON_RESIDUAL
  return max(vec4(0.0),texture(sampler,tc));
#else
  return texture(sampler,tc);
#endif
}
#endif

#ifdef USE_RESIDUAL
#ifdef BATCHNORM_ON_RESIDUAL
void handleResidual() {
  fragmentColor0 += batchnorm[0] * resFetch(resLayer0,resCoord);
#if NUM_LANES > 1
  fragmentColor1 += batchnorm[1] * resFetch(resLayer1,resCoord);
#endif
#if NUM_LANES > 2
  fragmentColor2 += batchnorm[2] * resFetch(resLayer2,resCoord);
#endif
#if NUM_LANES > 3
  fragmentColor3 += batchnorm[3] * resFetch(resLayer3,resCoord);
#endif
#if NUM_LANES > 4
  fragmentColor4 += batchnorm[4] * resFetch(resLayer4,resCoord);
#endif
#if NUM_LANES > 5
  fragmentColor5 += batchnorm[5] * resFetch(resLayer5,resCoord);
#endif
#if NUM_LANES > 6
  fragmentColor6 += batchnorm[6] * resFetch(resLayer6,resCoord);
#endif
#if NUM_LANES > 7
  fragmentColor7 += batchnorm[7] * resFetch(resLayer7,resCoord);
#endif
}
#else
void handleResidual() {
  fragmentColor0 += resFetch(resLayer0,resCoord);
#if NUM_LANES > 1
  fragmentColor1 += resFetch(resLayer1,resCoord);
#endif
#if NUM_LANES > 2
  fragmentColor2 += resFetch(resLayer2,resCoord);
#endif
#if NUM_LANES > 3
  fragmentColor3 += resFetch(resLayer3,resCoord);
#endif
#if NUM_LANES > 4
  fragmentColor4 += resFetch(resLayer4,resCoord);
#endif
#if NUM_LANES > 5
  fragmentColor5 += resFetch(resLayer5,resCoord);
#endif
#if NUM_LANES > 6
  fragmentColor6 += resFetch(resLayer6,resCoord);
#endif
#if NUM_LANES > 7
  fragmentColor7 += resFetch(resLayer7,resCoord);
#endif
}
#endif
#endif  // USE_RESIDUAL
