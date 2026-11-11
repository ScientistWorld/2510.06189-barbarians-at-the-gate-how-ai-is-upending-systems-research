# Problem Definition

## Research Question

Can a rebalancing algorithm for Mixture-of-Experts (MoE) expert-parallel inference do less work per rebalance while keeping GPU load just as balanced? This gym poses the Expert Parallelism Load Balancing (EPLB) problem from the paper's MoE case study, on the paper's own expert-load trace.

Every 100 inference steps the serving system hands the balancer the per-expert token load of the window that just ended, for every MoE layer. The balancer must return, for each layer:

- how many physical replicas each logical expert gets (exactly `num_replicas` slots, every expert at least one), and
- which physical slot (and therefore which GPU and node) holds each replica.

The cluster is hierarchical: 256 logical experts in 8 expert groups, 288 physical slots, 4 nodes with 8 GPUs each (9 slots per GPU). Each expert group must live entirely on one node, and every node hosts the same number of groups. The placement is then used for the next 100 steps, so it is judged on the next window's load, which the balancer has not seen.

## Why It Matters

When popular experts pile up on a few GPUs, the slowest GPU sets the latency of every MoE layer. Expert loads drift with the workload (prefill bursts, finishing requests, dataset changes), so placements must be recomputed often. The rebalancing computation runs inside the serving system; a slow balancer either delays rebalancing or burns serving time.

## Success Criteria

A good balancer:

- returns a valid placement for every layer of every window,
- keeps next-window GPU balance (mean GPU load / max GPU load) at least as good as the frontier-lab reference policy, and
- needs less rebalance work than that reference.

Better next-window balance (for example by anticipating load drift) is also a genuine improvement, as long as validity holds.
