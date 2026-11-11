# Paper's Method

## Key Contribution

The paper argues for AI-Driven Research for Systems (ADRS): an LLM-driven evolutionary loop (OpenEvolve) proposes code, and a reliable simulator scores it. In the Expert Parallelism Load Balancer (EPLB) case study (Section 5.2), the loop starts from the public DeepSeek/vLLM EPLB program. It discovers a rebalancing algorithm that matches the baselines' imbalance factor (0.66) while cutting rebalance time from 19.6 ms (a non-public frontier-lab reference) to 3.7 ms. The public greedy initial program takes 540 ms.

## The Three Policies Shipped Here

All three implement `rebalance_experts` with the same three-stage hierarchical structure:
1. pack expert groups onto nodes,
2. replicate the most overloaded experts within each node,
3. pack physical replicas onto the GPUs of each node.

- `baseline/eplb_greedy_deepseek.py` (`greedy_deepseek`): the paper's initial program, copied verbatim from the released case study. Both packing stages use Python-loop greedy bin packing (sort, then put each item into the least-loaded non-full pack).
- `baseline/eplb_frontier_reference.py` (`frontier_reference_zigzag`): reconstruction of the frontier-lab reference. The paper describes it as replacing the loop-based packing of replicas onto GPUs with a tensorised zigzag ("snake") layout that interleaves heavy and light replicas across GPUs, with the same imbalance factor. Here stage 3 uses the tensorised zigzag; stage 1 keeps the greedy loop and stage 2 the replication loop. The hidden test measures every speedup against a frozen copy of this file in `eval/`.
- `method/eplb_evolved.py` (`evolved_zigzag`): the reproduced evolved policy. Following Figure 5b, it sorts once, then within even blocks of `num_packs` items assigns packs in order and within odd blocks in reverse (`torch.where`). This happens without any Python loop, batched over all layers. As the paper reports for the breakthrough, the zigzag partitioning is reused in both packing stages, not just one. The replication stage keeps the rule of repeatedly replicating the currently most overloaded expert ("replicating only overloaded experts prevailed").

The paper's evolved code is not released, and its unspecified "subtle enhancements" (ordering logic, adaptive reshaping) are not reproduced. The shipped method is exactly the described change: zigzag in both stages, batched over layers.

## Main Claims Tested

- Primary: the evolved policy needs much less rebalance work than the frontier reference (speedup > 1).
- Constraint: next-window GPU balance stays the same as the frontier reference (and the greedy initial program), at the paper's 0.66 level.
