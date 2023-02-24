#ifdef USE_RESIDUAL
vec4 residual(in sampler2D sampler,in vec2 tc) {
#ifdef RELU_ON_RESIDUAL
  return max(vec4(0.0),texture(sampler,tc));
#else
  return texture(sampler,tc);
#endif
}
#endif

