# Caching Strategies

A cache stores the results of expensive operations closer to where they are needed, trading memory and the risk of staleness for lower latency and reduced load on the source of truth.

The most common pattern is cache-aside, also called lazy loading. The application first checks the cache; on a miss it reads from the database, then writes the result into the cache for next time. The cache only ever holds data that has actually been requested, but every miss costs an extra round trip.

With write-through caching, every write goes to the cache and the database synchronously, so the cache is never stale, but write latency increases. With write-back caching, also called write-behind, writes go only to the cache and are flushed to the database asynchronously, which makes writes fast but risks losing data if the cache node fails before flushing.

When a cache is full, an eviction policy decides what to remove. Least recently used, or LRU, evicts the entry that has gone longest without being accessed and is typically implemented with a hash map combined with a doubly linked list, giving constant-time lookups and updates. Least frequently used, or LFU, evicts entries with the fewest accesses and resists one-off scans better, while time-to-live expiry removes entries after a fixed age regardless of use.

A cache stampede, also known as the thundering herd problem, happens when a popular key expires and many concurrent requests miss at once and all hit the database. Common defenses are request coalescing, where only one request recomputes the value while others wait, and adding random jitter to expiration times so that keys do not expire simultaneously.
