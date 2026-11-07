# Raft Consensus

Raft is a consensus algorithm designed to be easier to understand than Paxos while providing the same safety guarantees. A Raft cluster replicates a log of commands across a set of servers so that every server applies the same commands in the same order, which lets a group of machines behave like a single reliable state machine.

At any moment each server is in one of three states: leader, follower, or candidate. In normal operation there is exactly one leader and all other servers are followers. Followers are passive: they only respond to requests from the leader and from candidates.

Time is divided into terms of arbitrary length, numbered with consecutive integers. Each term begins with an election. If a follower receives no communication from a leader during its election timeout, it assumes there is no viable leader, increments its current term, transitions to the candidate state, and requests votes from the other servers. Election timeouts are chosen randomly from a fixed interval, such as 150 to 300 milliseconds, so that split votes are rare and resolved quickly.

A candidate wins an election if it receives votes from a majority of the servers in the full cluster for the same term. Each server votes for at most one candidate per term, on a first-come-first-served basis. The majority rule ensures that at most one candidate can win the election for a particular term.

Once elected, the leader accepts client commands, appends them to its own log, and sends AppendEntries RPCs to the followers. An entry is considered committed once the leader has replicated it on a majority of servers. The leader also sends empty AppendEntries messages as heartbeats to maintain its authority and prevent new elections.

Raft restricts which servers may become leader: a candidate's vote request includes information about its log, and a voter denies its vote if its own log is more up-to-date than the candidate's. This election restriction guarantees that every newly elected leader already holds all committed entries, so committed data never flows from followers back to the leader.

A cluster of five servers can tolerate the failure of two servers, because the remaining three still form a majority.
