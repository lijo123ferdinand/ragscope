# Multi-Version Concurrency Control

Multi-version concurrency control, or MVCC, lets a database serve reads without blocking writes by keeping several versions of each row. Instead of overwriting a row, a writer creates a new version, and each transaction sees a consistent snapshot of the data as of a particular point in time.

The key benefit is that readers never block writers and writers never block readers. A long-running analytical query can scan a table while other transactions keep updating it, because the query keeps reading the versions that were visible when its snapshot was taken.

In PostgreSQL, every row version carries the ID of the transaction that created it, stored in a field called xmin, and the ID of the transaction that deleted or replaced it, stored in xmax. A transaction's snapshot records which transaction IDs had committed when it started, and a row version is visible only if its creator committed before the snapshot and its deleter did not.

Old row versions that no transaction can still see are dead tuples, and they waste space until they are reclaimed. PostgreSQL removes them with a background process called VACUUM; if vacuuming falls behind, tables bloat and queries slow down. Other engines such as MySQL's InnoDB keep old versions in a separate undo log and reconstruct earlier versions on demand.

Snapshot isolation, the isolation level that MVCC naturally provides, prevents dirty reads and non-repeatable reads, but it still permits an anomaly called write skew. In write skew, two transactions read overlapping data, make decisions based on what they read, and write to different rows, producing a result that no serial order would allow. PostgreSQL's serializable level detects these dangerous patterns and aborts one of the transactions.
