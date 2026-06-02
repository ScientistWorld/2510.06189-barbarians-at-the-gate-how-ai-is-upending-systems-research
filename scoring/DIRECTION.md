# Research Direction

## Research Problem

Improve expert-parallel load balancing for MoE inference under changing expert popularity. The central question is how to place replicated experts across GPUs with low rebalancing work while preserving balanced token load.

## Core Contribution

Work within the EPLB policy-design problem: deciding replica counts and physical placement for logical experts under fixed traces and cluster dimensions.

## Approach Scope

Improvements should modify the placement or replication policy for the same EPLB interface. Candidate policies may use different deterministic heuristics, vectorized layouts, local repair steps, or search over placements, as long as they output the same artifact format.

## Out of Bounds

- Replacing the task with MoE router training or changing expert popularity upstream.
- Changing the evaluation to use wall-clock runtime instead of operation-count work.
- Adding more GPUs, replicas, or traces to make the benchmark easier.
- Skipping validity or balance constraints to optimize only operation count.
