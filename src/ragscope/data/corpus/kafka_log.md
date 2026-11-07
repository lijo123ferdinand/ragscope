# Apache Kafka and the Distributed Log

Apache Kafka is a distributed event streaming platform built around a simple abstraction: an append-only, ordered log. Producers append records to the end of the log, and consumers read records sequentially, each tracking its own position, called the offset.

Records are organized into topics, and each topic is split into partitions. A partition is the unit of ordering and parallelism: records within a single partition are strictly ordered, but there is no ordering guarantee across partitions. Producers usually choose a partition by hashing the record key, so that all records with the same key land in the same partition and stay in order.

Consumers that share a group ID form a consumer group, and Kafka assigns each partition to exactly one consumer within the group. A group can therefore have at most as many active consumers as there are partitions; extra consumers sit idle. When consumers join or leave, the group rebalances and partitions are reassigned.

Each partition is replicated to several brokers. One replica is the leader and handles reads and writes, while followers copy its data. The set of replicas that are fully caught up is called the in-sync replica set, or ISR. A producer configured with acks set to all receives an acknowledgment only after every in-sync replica has the record, which together with a minimum in-sync replica setting protects against data loss when a broker fails.

Kafka retains records for a configured time or size rather than deleting them once they are consumed, so multiple independent consumers can read the same data and new consumers can replay history. Log compaction is an alternative retention mode that keeps at least the latest record for each key. Kafka also offers idempotent producers and transactions, which together enable exactly-once processing semantics within Kafka.
