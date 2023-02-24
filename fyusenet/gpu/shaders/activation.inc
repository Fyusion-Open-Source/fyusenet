// FIXME (mw) the condition compilation is messy and will block using
// different activations on different inputs (e.g. in concat or binary operators)
vec4 activate(in vec4 inpix) {
#ifndef NO_ACT
#ifdef ACT_RELU
#ifdef LEAKY_RELU
  vec4 sg = step(vec4(0),inpix);
  return (sg + (LEAKY_RELU * (vec4(1)-sg))) *inpix;
#else
  return max(vec4(0.0), inpix);
#endif // LEAKY_RELU
#endif // ACT_RELU
#ifdef ACT_CLIP
  return min(vec4(CLIP_HIGH), max(vec4(CLIP_LOW), inpix));
#endif // ACT_CLIP
#endif // NO_ACT
  return inpix;
}
