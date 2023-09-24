The input texture is assumed to be of the following form and the result of a single-token
dot-product operation done with the DotProdSingle class:

 @code
 +------+------+------+-----+------+
 | t0t0 | t0t1 | t0t2 | ... | t0tn | head0..3
 +------+------+------+-----+------+
 | t0t0 | t0t1 | t0t2 | ... | t0tn | head4..7
 +------+------+------+-----+------+
 |             ...                 |
 +------+------+------+-----+------+
 | t0t0 | t0t1 | t0t2 | ... | t0tn | head..k
 +------+------+------+-----+------+
@endcode

where \c t0t2 for example refers to the dot product of the query token 0 with the key token 2.

Internally a two-pass approach is used which first computes the denominators for the softmax
operation, these are stored in an intermediary texture as a single column in the format:
@code
 +-----------+
 | head 0..3 |
 +-----------+
 | head 4..7 |
 +-----------+
 |    ...    |
 +-----------+
 | head ..k  |
 +-----------+
@endcode

The 2nd pass then computes the softmax operation and writes the result to the target FBO, which
has the following format:

@code
   RGBA   RGBA   RGBA         RGBA
 +------+------+------+-----+------+
 | t0t0 | t0t1 | t0t2 | ... |   0  | head 0..3
 +------+------+------+-----+------+
 | t0t0 | t0t1 | t0t2 | ... |   0  | head 4..7
 +------+------+------+-----+------+
 |                ...              |
 +------+------+------+-----+------+
 | t0t0 | t0t1 | t0t2 | ... |   0  | head k-3..k
 +------+------+------+-----+------+
@endcode

where we highlighted some causal masking on the right end of the output (which is usually not
the case).

