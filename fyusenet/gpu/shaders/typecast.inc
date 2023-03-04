// NOTE (mw) this does not take into account any precision issues in storage
ivec4 castTo(in vec4 input) {
#ifdef CAST_TO_INT32
    return max(-2147483648,min(2147483647,round(input)));
#endif
#ifdef CAST_TO_INT16
    return max(-32768,min(32767,round(input)));
#endif
#ifdef CAST_TO_INT8
    return max(-128,min(127,round(input)));
#endif
#ifdef CAST_TO_UINT32
    return max(0,min(4294967295,round(input)));
#endif
#ifdef CAST_TO_UINT16
    return max(0,min(65535,round(input)));
#endif
#ifdef CAST_TO_UINT8
    return max(0,min(255,round(input)));
#endif
    return ivec4(input);
}
