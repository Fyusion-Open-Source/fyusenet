The input for this shader (and the related fragment shader) is a set of proxy polygons and a
total of two textures, representing the attention weights and the value tensor. The attention
weights are represented by a single (causally-masked) tile, as is shown in the example below for
n keys:

@code
      RGBA   RGBA
    +------+------+------+-----+------+
    | t0t0 | t0t1 | t0t2 | ... | t0tn | (head 0..3)
    +------+------+------+-----+------+
    | t0t0 | t0t1 | t0t2 | ... | t0tn | (head 4..7)
    +------+------+------+-----+------+
    |             ...                 |
    +------+------+------+-----+------+
    | t0t0 | t0t1 | t0t2 | ... | t0tn | (head k-3 .. k)
    +------+------+------+-----+------+
@endcode

When causal masking is in play, the texture may look like this:

@code
    +------+------+------+-----+------+
    | t0t0 | t0t1 | t0t2 | ... |  00  | (head 0..3)
    +------+------+------+-----+------+
    | t0t0 | t0t1 | t0t2 | ... |  00  | (head 4..7)
    +------+------+------+-----+------+
    |             ...                 |
    +------+------+------+-----+------+
    | t0t0 | t0t1 | t0t2 | ... |  00  | (head k-3 .. k)
    +------+------+------+-----+------+
@endcode

The value tensor is provided as a single texture with a row-by-row layout, where one row
corresponds to one query, and each row is tiled with the number of heads. Note that we make
use of caching for the key and the value data, therefore the number of rows in the value
tensor is always equivalent to the number of keys. An example value texture may look like
this (n keys, k heads, head dimension d, with d=64 and k=16):

@code
       m pixels      m pixels                                m pixels
    +-------------+-------------+-----------------------+-----------------+
    | T0(0) head0 | T0(d) head1 | ......................| T0(1023) head k |
    +-------------+-------------+-----------------------+-----------------+
    | T1(0) head0 | T0(d) head1 | ......................| T0(1023) head k |
    +-------------+-------------+-----------------------+-----------------+
    |     ...     |     ...     |         ...           |      ...        |
    +-------------+-------------+-----------------------+-----------------+
    | Tn(0) head0 | Tn(d) head1 | ......................| Tk(1023) head k |
    +-------------+-------------+-----------------------+-----------------+
@endcode

The result of the multiplication between the attention weights and the value tensor is another
tensor of size #heads x headdim x 1 x #values which is stored as single row in the output
texture as follows:

@code
    +-------------+-------------+-----------------------+---------------+
    | T0(0) head0 | T0(d) head1 | ......................| T0(??) head k |
    +-------------+-------------+-----------------------+---------------+
@endcode

The multiplication is done by using a set of line segments as proxy geometry and also use
instanced rendering for the "inner loops" to (hopefully) keep the SMs more busy.
