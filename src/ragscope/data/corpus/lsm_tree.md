# Log-Structured Merge Trees

A log-structured merge tree, or LSM tree, is a storage structure optimized for write-heavy workloads. Instead of updating data in place, it buffers writes in memory and later flushes them to disk as large sequential files, turning random writes into sequential ones.

Incoming writes are first appended to a write-ahead log for durability and then inserted into an in-memory sorted structure called the memtable. When the memtable reaches a configured size, it is frozen and written to disk as an immutable sorted string table, commonly called an SSTable.

Because SSTables are never modified after they are written, an update to a key simply writes a newer version, and a delete writes a special marker known as a tombstone. Old versions and tombstones are physically removed later.

Reads must consult the memtable first and then the SSTables from newest to oldest, since the newest version of a key wins. To avoid opening every file, each SSTable usually carries a Bloom filter that can quickly rule out files that definitely do not contain the requested key.

Over time the number of SSTables grows, so a background process called compaction merges overlapping files, discards overwritten values and expired tombstones, and writes out fewer, larger files. Size-tiered compaction merges files of similar size and favors write throughput, while leveled compaction organizes files into levels with non-overlapping key ranges and favors read performance and space efficiency.

The main costs of the design are amplification effects. Write amplification is the ratio of bytes written to storage versus bytes written by the application, and it rises because compaction rewrites the same data several times. Read amplification counts how many files a lookup may need to check. Space amplification measures how much extra disk is consumed by stale versions that have not yet been compacted away.

LSM trees are the storage engine behind systems such as LevelDB, RocksDB, Cassandra, and HBase.
