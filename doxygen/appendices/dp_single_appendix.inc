The shader used in this operator takes as input two textures (Q, K) with identical size using the
following format:
     - one row per token
     - row length equivalent to #heads x head dimension (divided by 4 due to RGBA pixels)
For example, a set of k tokens with 32 heads and a head dimension of 128 would result in the
following layout:

@code
 32 (head_dim/4) 32 (head_dim/4)                           32 (head-dim/4)
+---------------+---------------+-----------------------+------------------+
|  T0(0) head0  | T0(32) head1  | ......................| T0(1023) head 32 |
+---------------+---------------+-----------------------+------------------+
|  T1(0) head0  | T1(32) head1  | ......................| T1(1023) head 32 |
+---------------+---------------+-----------------------+------------------+
|      ...      |      ...      |         ...           |      ...         |
+---------------+---------------+-----------------------+------------------+
|  Tk(0) head0  | Tk(32) head1  | ......................| Tk(1023) head 32 |
+---------------+---------------+-----------------------+------------------+
@endcode

The output of the shader will be arranged as shown below, where each batch of 4 heads occupies
a single row in the texture (RGBA texture layout is used):

@code
+------+------+------+-----+------+
| t0t0 | t0t1 | t0t2 | ... | t0tn | head(0..3)
+------+------+------+-----+------+
| t0t0 | t0t1 | t0t2 | ... | t0tn | head(4..7)
+------+------+------+-----+------+
|             ...                 |
+------+------+------+-----+------+
| t0t0 | t0t1 | t0t2 | ... | t0tn | head(k-3 .. k)
+------+------+------+-----+------+
@endcode

For example, \c t0t0 corresponds to the inner product between query token 0 and key token 0
and \c t0t2 corresponds to the inner product between query token 0 and key token 2. The
batching into 4 heads per result is done due to the RGBA pixel storage format.
