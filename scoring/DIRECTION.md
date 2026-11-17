# Research Direction

## Research Problem

Design expert-parallel load-balancing policies for MoE inference. They must rebalance cheaply when expert popularity shifts, while keeping the most loaded GPU close to the average on the traffic that follows.

## Core Contribution

Work inside the EPLB policy-design problem: choosing replica counts and the hierarchical placement (groups to nodes, replicas to GPUs) from the last window's per-expert load.

## Approach Scope

Two kinds of improvement count:
- doing the same placement work with fewer executed instructions (vectorisation, fewer passes over the data, fewer library calls, cheaper replication or packing steps, smaller outputs);
- placing replicas so that balance on the next, unseen window improves (for example by making the placement robust to load drift between windows).

Both must keep placements valid and balance no worse than the reference.

## Out of Bounds

- Using the held-out layers or time blocks or the raw trace, or memorising visible windows.
- Changing the cluster configuration, the windowing, or the evaluator and its measurement.
- Trying to escape or tamper with the instruction count or the sandbox, or self-reporting work.
- Trading away validity or balance for speed.
