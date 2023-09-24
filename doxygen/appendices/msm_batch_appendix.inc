As the shaders in this class run "batch-wise", we expect input textures of the form:
@code
    RGBA    RGBA    RGBA        RGBA
   +------+------+------+-----+------+
   | t0t0 | t0t1 | t0t2 | ... | t0tn |   \
   +------+------+------+-----+------+    \
   | t1t0 | t1t1 | t1t2 | ... | t1tn |     \
   +------+------+------+-----+------+      --> heads i..i+3
   |             ...                 |     /
   +------+------+------+-----+------+   /
   | tmt0 | tmt1 | tmt2 | ... | tmtn | /
   +------+------+------+-----+------+
@endcode
where \c t1t2 refers to the dot product of query token 1 with query token 2. Each entry stores
the result for 4 heads. It is possible to stack these result blocks vertically in case more than
one set of heads is to be processed in a single batch.

The denominators computed in the first pass using vertical lines as geometry proxies and also
employing instanced rendering for accumulating the results in <i>inner batches</i>.
For example, when operating on a tensor given by querylen=20 and keylen=24 and an inner batch
size of 4, a total of:
@code
                           ceil(keylen/bs) = ceil(24/4) = 6
@endcode
instances is required to complete the norm computation. The result will be a texture with a single
column of the form:

@code
    +----+
    | t0 |
    +----+
    | t1 |
    +----+ (4 heads per column as RGBA pixels)
    | .. |
    +----+
    | tm |
    +----+
@endcode

The final pass will use the dot-product input and the computed denominators to compute the final
(masked) softmax into the following output format:

@code
          <---  keyspace  --->
    +------+------+------+-----+------+
    | t0t0 |  00  |  00  | ... |  00  | q0
    +------+------+------+-----+------+
    | t1t0 | t1t1 |  00  | ... |  00  | q1
    +------+------+------+-----+------+  (4 heads per tile as RGBA pixels)
    |             ...                 |
    +------+------+------+-----+------+
    | tmt0 | tmt1 | tmt2 | ... | tmtn | qm
    +------+------+------+-----+------+
@endcode

where we indicated some masking in the example.
