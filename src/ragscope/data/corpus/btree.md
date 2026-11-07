# B-Trees and B+ Trees

A B-tree is a self-balancing search tree designed for storage systems that read and write data in large blocks. Each node holds many keys and has many children, so the tree stays very shallow even when it indexes billions of records. A high branching factor means that a lookup touches only a handful of disk pages.

All leaves of a B-tree sit at the same depth. When an insertion overflows a node, the node is split into two and the middle key is pushed up into the parent; if the parent overflows as well, the split propagates upward, and the tree only grows in height when the root itself splits.

Most relational databases actually use a variant called the B+ tree. In a B+ tree, internal nodes store only keys used for routing, while all records or record pointers live in the leaf nodes. The leaves are linked together in a sorted chain, which makes range scans efficient: the engine descends once to the first matching leaf and then follows sibling pointers.

Because B-trees update pages in place, a crash in the middle of a page split could leave the tree corrupted. Databases therefore protect B-tree modifications with a write-ahead log, recording each change before applying it to the page so that recovery can redo or undo incomplete operations.

Compared with LSM trees, B-trees generally offer more predictable read latency, since each key lives in exactly one place, but they perform random writes and can suffer from page fragmentation. PostgreSQL, MySQL's InnoDB engine, and SQLite all rely on B-tree indexes.
