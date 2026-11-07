# Two-Phase Commit

Two-phase commit, or 2PC, is a protocol for making a transaction that spans several databases or services either commit everywhere or abort everywhere. One node acts as the coordinator and the others act as participants.

In the first phase, the prepare phase, the coordinator asks every participant whether it can commit. Each participant does all the work needed to guarantee it can commit, writes its changes and a prepare record to durable storage, and votes yes or no. A participant that votes yes gives up the right to abort on its own.

In the second phase, the commit phase, the coordinator decides. If every participant voted yes, the coordinator writes a commit decision to its log and tells everyone to commit; if any participant voted no or failed to answer, the coordinator tells everyone to abort. The decision written to the coordinator's log is the point of no return.

The weakness of two-phase commit is blocking. If the coordinator crashes after participants have voted yes but before they learn the decision, those participants are stuck in doubt: they cannot commit or abort unilaterally and must hold their locks until the coordinator recovers. Three-phase commit tries to avoid this but assumes bounded network delays, so practical systems instead make the coordinator highly available, for example by replicating its log with a consensus protocol, as Google Spanner does with Paxos.

Because of these costs, many microservice architectures avoid distributed transactions altogether and use the saga pattern, in which a long-running business transaction is split into a sequence of local transactions, each with a compensating action that undoes its effect if a later step fails.
