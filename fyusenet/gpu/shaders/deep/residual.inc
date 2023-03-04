#ifdef USE_RESIDUAL
vec4 residual(in sampler2D sampler, in vec2 tc, in sampler2D biasTex, in vec2 bc) {
  vec4 res;
#ifdef RELU_ON_RESIDUAL
  res = max(vec4(0.0), texture(sampler,tc));
#else
  res = texture(sampler,tc);
#endif
#ifdef BATCHNORM_ON_RESIDUAL
  res = applyBNNB(res, biasTex, ivec4(bc, 0, 1));
#endif
  return res;
}
#endif
