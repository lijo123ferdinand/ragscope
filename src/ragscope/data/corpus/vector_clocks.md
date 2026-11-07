# Vector Clocks

In a distributed system there is no single global clock that all nodes agree on, so ordering events requires logical clocks. A Lamport clock assigns each event a single counter that increases with every local event and message, which guarantees that if event A happened before event B then A's timestamp is smaller. The converse does not hold: a smaller Lamport timestamp does not prove that one event caused another.

A vector clock fixes this limitation by keeping one counter per node. Each node increments its own entry before every local event, attaches the full vector to every message it sends, and on receiving a message takes the element-wise maximum of its own vector and the received one before incrementing its own entry.

Two vector clocks can be compared entry by entry. Clock A happened before clock B if every entry of A is less than or equal to the corresponding entry of B and at least one entry is strictly smaller. If neither clock happened before the other, the events are concurrent, meaning neither could have caused the other.

Detecting concurrency is exactly what replicated data stores need. Amazon's Dynamo attached vector clocks to object versions so that when two clients updated the same key on different replicas, the system could recognize the conflicting siblings and return both to the application for reconciliation, instead of silently discarding one with a last-writer-wins rule.

The main drawback is size: a vector clock grows with the number of nodes that have ever written the item. Practical systems truncate old entries or use variants such as dotted version vectors, which track causality per replica rather than per client.
