# TCP Congestion Control

TCP congestion control prevents senders from overwhelming the network. Besides the receive window advertised by the receiver, every TCP sender maintains a congestion window, which limits how many bytes may be in flight before an acknowledgment arrives. The effective sending window is the minimum of the two.

A new connection starts in slow start. The congestion window begins small, typically ten segments on modern systems, and grows by one segment for every acknowledgment received, which doubles the window every round-trip time. Despite its name, slow start therefore grows exponentially.

When the window reaches the slow start threshold, the sender switches to congestion avoidance and grows the window by roughly one segment per round-trip time. This additive increase, combined with multiplicative decrease on loss, is known as AIMD, and it allows competing flows to converge toward a fair share of the bottleneck link.

TCP Reno treats packet loss as the signal of congestion. If the sender receives three duplicate acknowledgments, it performs fast retransmit, resending the missing segment without waiting for a timeout, and fast recovery, halving the congestion window instead of collapsing it. A retransmission timeout is treated as a more severe signal and resets the window to one segment.

CUBIC, the default algorithm in Linux, grows the window as a cubic function of the time since the last loss, which makes it scale better on long, high-bandwidth paths. Google's BBR takes a different approach: rather than reacting to loss, it builds a model of the path's bottleneck bandwidth and round-trip propagation time and paces packets to match the estimated bandwidth, which avoids filling router buffers and reduces queuing delay.
