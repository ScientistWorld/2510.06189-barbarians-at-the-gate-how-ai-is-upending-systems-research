# Paper's Method

## Key Contribution

The paper argues for AI-Driven Research for Systems (ADRS): use LLM-driven iterative code generation plus reliable simulators to discover systems algorithms. The reproduced case study is Expert Parallelism Load Balancing (EPLB) for MoE inference.

## Approach

For EPLB, the algorithmic pipeline has three stages: distribute expert groups across nodes, create replicas for overloaded experts, and assign physical replicas to GPUs. The initial program uses greedy bin packing. The evolved policy keeps the overloaded-expert replication idea but replaces loop-heavy placement with a PyTorch tensorized snake/zigzag assignment that sorts replicas and alternates pack direction across blocks. This reproduction makes the group-to-node zigzag order feed into physical replica materialization, then applies the same tensorized placement at the GPU stage. The paper emphasizes that applying this zigzag partitioning across multiple EPLB stages was the key improvement.

## Main Claims

- Primary: ADRS can discover a faster EPLB rebalancing algorithm without worsening load balance.
- Reproduced target: the evolved zigzag placement reduces rebalance work relative to greedy while keeping valid, similarly balanced placements.
- Secondary paper-level claim: reliable evaluators make systems algorithm discovery especially suitable for AI-driven search.
