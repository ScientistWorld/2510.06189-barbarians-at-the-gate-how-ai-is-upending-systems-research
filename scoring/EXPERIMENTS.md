# Experiment Overview

## eplb_moe_placement

Establishes whether an expert-parallel load balancer for MoE inference can rebalance with less computation without losing GPU load balance. The setting is the expert-load trace of a real MoE serving run (ShareGPT and GSM8K traffic), rebalanced on 100-step windows on a hierarchical cluster (288 expert slots, 4 nodes, 32 GPUs, expert groups kept within one node).

- Benefit: how much less rebalance work a policy needs than a fixed frontier-lab reference policy. The evaluator counts the CPU instructions each rebalance executes, from outside the policy's process, on the same windows for both (`speedup_vs_frontier_reference`).
- Constraint: GPU load balance on the window after each rebalance, which the policy has not seen (`balancedness`). It must not fall below the reference's. Every placement must also be valid: exact replica budget, every expert deployed, consistent maps, groups kept whole on equally loaded nodes. Any failure scores the run as failed.

The hidden test uses MoE layers of the same trace that are never visible, in time periods that no visible layer covers, with a secret relabelling of expert ids. A policy must therefore work on load distributions and moments of the serving run it has not seen. It is never scored on the window it was given.
