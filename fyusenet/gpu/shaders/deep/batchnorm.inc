#ifdef POST_BATCHNORM
vec4 applyBN(in vec4 col, in sampler2D bsSampler,in ivec4 chans) {
#ifdef NO_BIAS
  return col*texelFetch(bsSampler,chans.yw,0);
#else
  return col*texelFetch(bsSampler,chans.yw,0)+texelFetch(bsSampler,chans.xz,0);
#endif
}

vec4 applyBNNB(in vec4 col, in sampler2D bsSampler, in ivec4 chans) {
  return col*texelFetch(bsSampler,chans.yw,0);
}
#endif

