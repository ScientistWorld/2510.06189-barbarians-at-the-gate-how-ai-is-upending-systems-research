# Scientist Constraints

## Data

Use only the visible pairs in `data/eplb_train_windows.json` (even MoE layers of the paper's trace, time blocks 0 and 2) for development. The hidden test uses the other layers in the other time blocks. Do not download, embed or otherwise use the raw `expert-load.json` trace, any held-out layer or any held-out time block in a method.

## Problem Size

The cluster configuration is fixed: 256 logical experts, 8 expert groups, 288 physical slots, 4 nodes, 32 GPUs, 100-step windows, scoring on the next window. A method must accept these as arguments and must not change them.

## Interface and Measurement

- A method is one self-contained policy file `outputs/eplb/<name>.py` (at most 64 KB) defining `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus) -> (phy2log, log2phy, logcnt)`.
- It may import `torch`, `numpy` and the standard library. It runs in a sandbox without file, network or exec access; every instruction it executes during a call, in any thread or process it starts, counts as rebalance work.
- It must be deterministic and must not try to interfere with the evaluator, its measurement or its sandbox.
- Score with `scripts/evaluate_train.sh` / `scripts/evaluate.sh` unchanged. Do not edit `eval/`, the anchor policy, or the data files to change scores.

## Compliance Checklist

- [ ] Policy files are self-contained and deterministic
- [ ] Placements are valid for every window (budget, coverage, consistent maps, groups whole on equally loaded nodes)
- [ ] Balancedness is not below the frontier reference by more than 0.005
- [ ] No use of held-out layers or time blocks, the raw trace, or evaluator internals
- [ ] No wall-clock timing or self-reported work replaces the evaluator's instruction count
