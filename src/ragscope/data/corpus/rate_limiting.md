# Rate Limiting Algorithms

Rate limiting protects services from overload and abuse by capping how many requests a client may make in a period of time. The choice of algorithm determines how bursts are handled and how much state must be stored.

The token bucket algorithm keeps a bucket that holds up to a fixed capacity of tokens and refills at a constant rate. Each request removes one token, and a request that finds the bucket empty is rejected or delayed. Because the bucket can fill up while a client is idle, the token bucket permits short bursts up to the bucket capacity while still enforcing the long-term average rate.

The leaky bucket algorithm instead places requests in a queue that drains at a constant rate. It smooths traffic into a steady output stream, which is useful for protecting a downstream system, but it does not allow bursts to be processed faster than the drain rate.

A fixed window counter counts requests in discrete intervals such as each minute. It is cheap, needing a single counter per client, but it suffers from a boundary problem: a client can send a full quota at the end of one window and another full quota at the start of the next, doubling the intended rate for a short time. The sliding window log stores the timestamp of every request and is exact but memory-hungry, while the sliding window counter approximates it by weighting the previous window's count by how much of it overlaps the current sliding interval.

In a distributed deployment the limiter's state must be shared, commonly in Redis, using atomic operations or Lua scripts so that concurrent requests on different servers do not race. When a request is rejected, HTTP APIs conventionally respond with status code 429 Too Many Requests and a Retry-After header.
