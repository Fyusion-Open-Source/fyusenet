#ifdef POST_BATCHNORM
vec4 applyBN(in vec4 col, in sampler2D sampler,in ivec4 chans) {
#ifdef NO_BIAS
  return col*texelFetch(sampler,chans.yw,0);
#else
  return col*texelFetch(sampler,chans.yw,0)+texelFetch(sampler,chans.xz,0);
#endif
}
#endif

