#ifndef NO_HALF
vec4 compute(in vec4 tex,in int offset) {
  mediump mat4 weights;
  tex = activate(tex);
  highp uvec4 w = layer0coeffs[offset];
  weights[0] = vec4(unpackHalf2x16(w.x),unpackHalf2x16(w.y));
  weights[1] = vec4(unpackHalf2x16(w.z),unpackHalf2x16(w.w));
  w = layer0coeffs[offset+1];
  weights[2] = vec4(unpackHalf2x16(w.x),unpackHalf2x16(w.y));
  weights[3] = vec4(unpackHalf2x16(w.z),unpackHalf2x16(w.w));
  return tex*weights;
}
#else
vec4 compute(in vec4 tex,in int offset) {
  mediump mat4 weights;
  tex = activate(tex);
  weights[0] = layer0coeffs[2*offset];
  weights[1] = layer0coeffs[2*offset+1];
  weights[2] = layer0coeffs[2*offset+2];
  weights[3] = layer0coeffs[2*offset+3];
  return tex*weights;
}
#endif
