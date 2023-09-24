As with the single token dot product computation, the input for the shader is provided by two
textures (Q,K)  using the following format:
     - one row per token
     - row length equivalent to #heads x head dimension (divided by 4 due to RGBA pixels)
For example, a set of k tokens with 32 heads and a head dimension of 128 would result in the
following layout:

@code
 32 (head_dim/4) 32 (head_dim/4)                          32 (head-dim/4)
+---------------+---------------+-----------------------+------------------+
|  T0(0) head0  | T0(32) head1  | ..................... | T0(1023) head 32 |
+---------------+---------------+-----------------------+------------------+
|  T1(0) head0  | T1(32) head1  | ..................... | T1(1023) head 32 |
+---------------+---------------+-----------------------+------------------+
|      ...      |      ...      |          ...          |       ...        |
+---------------+---------------+-----------------------+------------------+
|  Tk(0) head0  | Tk(32) head1  | ..................... | Tk(1023) head 32 |
+---------------+---------------+-----------------------+------------------+
@endcode

As we are computing the dot-product in batches along the head-dimension, we store the result
in an RGBA texture with a batch size that is a multiple of 4 and the output of the dot-product
computation for a single 4-element head batch looks like this:

@code
+------+------+------+-----+------+
| t0t0 | t0t1 | t0t2 | ... | t0tn |
+------+------+------+-----+------+
| t1t0 | t1t1 | t1t2 | ... | t1tn |
+------+------+------+-----+------+
|             ...                 |
+------+------+------+-----+------+
| tmt0 | tmt1 | tmt2 | ... | tmtn |
+------+------+------+-----+------+
@endcode

Where \c m refers to the number of query tokens and \c n to the number of key tokens. For example
for a 2048 token sequence, the dot-product texture would be 2048 x 2048 (WxH) \e pixels in size
for a batch of 4 heads.

The computation uses instanced rendering as it processes the computation in "inner batches" to
(try to) distribute the workload better among the SMs. The host code controls the inner batch
size, which usually defaults to 4 (see the #innerBatchSize_ definition). To provide an example,
the number of instances required to compute the dot-product of a 8x32x128 tensor with another
8x32x128 tensor (assuming #heads = 32 and headdim = 128) using an inner batch size of 4 requires
headdim/(PIXEL_PACKING * BS) = 128/(4*4) = 8 instances.
