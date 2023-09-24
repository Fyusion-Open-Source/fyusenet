The input for the shaders used in this class consists of the attention weight texture which contains
one or several batches of attention weights, and the value tensor which contains the values (V matrix)
of the multi-head attention operation.

For a single batch, the attention weights may look as follows (m queries, n keys, some masking
applied already):

@code
   RGBA   RGBA                RGBA
 +------+------+------+-----+------+
 | t0t0 |  00  |  00  | ... |  00  |  (heads 0..3)
 +------+------+------+-----+------+
 | t1t0 | t1t1 |  00  | ... |  00 |   (heads 0..3)
 +------+------+------+-----+------+  (4 heads as RGBA pixels)
 |             ...                 |
 +------+------+------+-----+------+
 | tmt0 | tmt1 | tmt2 | ... | tmtn |  (heads 0..3)
 +------+------+------+-----+------+
@endcode

The value tensor is provided as a single texture using a row-by-row layout, where one row corresponds
to one token, and each row is tiled with the number of heads.  An example value texture may look like
this (n keys, k heads, head dimension d, with d=64 and k=16):

@code
   m pixels       m pixels                               m pixels
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
tensor of size <tt>#heads x headdim x #queries x #values</tt> which is stored as a tiled row-by-row
arrangement in the output texture as follows (example for batch size of 1):

@code
 +-------------+-------------+-----+---------------+
 | T0(0) head0 | T0(d) head1 | ... | T0(??) head 3 |
 +-------------+-------------+-----+---------------+
 | T1(0) head0 | T0(d) head1 | ... | T0(??) head 3 |
 +-------------+-------------+-----+---------------+
 |     ...     |     ...     | ... |      ...      |
 +-------------+-------------+-----+---------------+
 | Tm(0) head0 | Tm(d) head1 | ... | Tm(??) head 3 |
 +-------------+-------------+-----+---------------+
@endcode

Note that the width of the data in the output texture for a single batch if limited to 4 heads. With
each following batch, the width of the data in output texture will grow according to the batch size.
all batches have been completed, the resulting texture data will have full width.
