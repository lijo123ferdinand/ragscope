# Bloom Filters

A Bloom filter is a space-efficient probabilistic data structure that answers set membership queries. It can tell you that an element is definitely not in the set, or that it is possibly in the set. False positives are possible, but false negatives are not.

The filter consists of a bit array of m bits, all initially zero, and k independent hash functions. To add an element, the filter hashes it with each of the k functions and sets the corresponding k bits to one. To query an element, it checks the same k positions: if any bit is zero, the element was never added.

The false positive rate depends on the number of bits, the number of hash functions, and the number of inserted elements n. For a given ratio of bits to elements, the optimal number of hash functions is approximately (m / n) multiplied by the natural logarithm of 2. With about ten bits per element and the optimal number of hash functions, the false positive rate is roughly one percent.

A standard Bloom filter does not support deletion, because clearing a bit might also remove evidence of other elements that hashed to the same position. A counting Bloom filter replaces each bit with a small counter so that elements can be removed by decrementing counters, at the cost of several times more memory.

Bloom filters are widely used to avoid expensive lookups. Storage engines check them before reading an SSTable from disk, web caches use them to avoid caching one-hit wonders, and network systems use them to summarize the contents of a peer's cache.
