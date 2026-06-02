# Problem Definition

## Research Question

Can automated algorithm discovery improve small, performance-critical systems algorithms when a reliable simulator can score correctness and performance? This reproduction focuses on the MoE expert-parallel load-balancing task: given changing per-expert token loads, a fixed number of logical experts, physical expert replicas, nodes, and GPUs, produce a replica count and GPU placement that balances generated-token load across GPUs.

## Why It Matters

Mixture-of-Experts inference can become bottlenecked when popular experts concentrate on a subset of GPUs. A good load balancer lets serving systems adapt to workload shifts while keeping reconfiguration overhead small.

## Success Criteria

A successful solution produces valid placements for all traces, maintains high average/max GPU load balance, and reduces the amount of work needed to rebalance experts when loads change.
