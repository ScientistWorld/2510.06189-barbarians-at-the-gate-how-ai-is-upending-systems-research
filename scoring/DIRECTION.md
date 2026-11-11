# Research Direction

## Research Problem

Design expert-parallel load-balancing policies for MoE inference. They must rebalance cheaply when expert popularity shifts, while keeping the most loaded GPU close to the average on the traffic that follows.

## Core Contribution

Work inside the EPLB policy-design problem: choosing replica counts and the hierarchical placement (groups to nodes, replicas to GPUs) from the last window's per-expert load.

## Approach Scope

Two kinds of improvement count:
- doing the same placement work with fewer operations (vectorisation, fewer passes over the data, cheaper replication or packing steps);
- placing replicas so that balance on the next, unseen window improves (for example by anticipating load drift with state kept across rebalances of a sequence).

Both must keep placements valid and balance no worse than the reference.

## Out of Bounds

- Using the held-out layers or the raw trace, or memorising visible windows.
- Changing the cluster configuration, the windowing, or the evaluator and its work measure.
- Escaping the work measure (threads, processes, native code, TorchScript, disabling the meter) or self-reporting work.
- Trading away validity or balance for speed.
