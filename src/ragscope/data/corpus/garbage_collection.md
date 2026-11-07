# Garbage Collection

Garbage collection automatically reclaims memory occupied by objects that a program can no longer reach. The collector starts from a set of roots, such as global variables, thread stacks, and registers, and treats any object reachable from those roots as live; everything else is garbage.

Reference counting keeps a count of references to each object and frees the object as soon as the count drops to zero. It reclaims memory promptly and incrementally, but it cannot by itself collect cycles of objects that refer to each other, and updating counts adds overhead to every pointer write. CPython uses reference counting plus a separate cycle detector to handle this case.

Tracing collectors instead periodically discover live objects. A mark-and-sweep collector marks every object reachable from the roots and then sweeps the heap to free unmarked objects. A copying collector moves live objects into a fresh region, which compacts memory and makes allocation a cheap pointer bump, at the cost of reserving extra space.

Most production collectors are generational, based on the weak generational hypothesis: most objects die young. New objects are allocated in a young generation that is collected frequently and cheaply, and objects that survive several collections are promoted to an old generation that is collected less often. To collect the young generation without scanning the whole old generation, the runtime uses a write barrier to record pointers from old objects to young objects in a remembered set.

The central trade-off is between throughput and pause time. Stop-the-world collectors pause all application threads while they work. Concurrent collectors such as Java's G1 and ZGC, and Go's tri-color concurrent mark-and-sweep collector, do most of their work alongside the application; ZGC targets pause times below one millisecond even on very large heaps.
