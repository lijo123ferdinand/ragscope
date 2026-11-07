# Conflict-Free Replicated Data Types

Conflict-free replicated data types, known as CRDTs, are data structures that can be updated independently and concurrently on different replicas and are mathematically guaranteed to converge to the same state once every replica has received the same updates. They provide strong eventual consistency without requiring coordination or consensus at write time.

There are two broad families. State-based CRDTs, also called convergent replicated data types, periodically ship their whole state to other replicas, and a merge function combines states. For convergence, the merge must be commutative, associative, and idempotent, which means the states form a join-semilattice. Operation-based CRDTs instead broadcast each update as an operation, and they rely on a delivery layer that delivers every operation exactly once and in causal order.

The simplest example is the grow-only counter, or G-Counter. Each replica keeps a vector of counts with one slot per replica and only increments its own slot. The value of the counter is the sum of all slots, and merging two states takes the per-slot maximum. A PN-Counter supports decrements by combining two G-Counters, one for increments and one for decrements, and reporting their difference.

Sets are trickier, because an element may be added on one replica and removed on another at the same time. An observed-remove set, or OR-Set, tags every addition with a unique identifier and a removal only deletes the tags it has observed, so a concurrent add wins over a remove.

CRDTs power collaborative editors, offline-first mobile applications, and geo-replicated databases such as Riak and Redis Enterprise's active-active mode. Their cost is metadata overhead: tombstones and per-replica tags can grow without bound unless the system performs garbage collection.
