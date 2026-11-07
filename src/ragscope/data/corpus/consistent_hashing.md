# Consistent Hashing

Consistent hashing is a technique for distributing keys across a changing set of nodes so that adding or removing a node moves only a small fraction of the keys. With naive modulo hashing, where a key is assigned to node hash(key) mod N, changing N remaps almost every key, which would flush caches and trigger massive data movement.

In consistent hashing, both nodes and keys are hashed onto the same circular space, often called the hash ring. Each key is stored on the first node encountered when walking clockwise around the ring from the key's position. When a node joins, it takes over only the keys between itself and its predecessor; when a node leaves, its keys move to its successor.

On average, when a cluster grows from N to N plus one nodes, only about 1 divided by (N plus one) of the keys need to move.

With only one position per node, the ring tends to be unbalanced: some nodes own large arcs and receive far more keys than others. The standard fix is virtual nodes, where every physical node is placed on the ring at many positions, for example one hundred or more. Virtual nodes smooth out the load and also let a more powerful machine take a proportionally larger share by receiving more positions.

Systems such as Amazon's Dynamo and Apache Cassandra use consistent hashing to partition data, and they replicate each key to the next several distinct nodes clockwise on the ring. An alternative called rendezvous hashing, or highest random weight hashing, scores every node for each key and picks the highest, achieving a similar minimal-movement property without a ring.
