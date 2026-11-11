# Scientist Constraints

## Data

Use only the visible windows in `data/eplb_train_windows.json` (even MoE layers of the paper's trace) for development. The hidden test uses the other layers. Do not download, embed or otherwise use the raw `expert-load.json` trace or any held-out layer in a method.

## Problem Size

The cluster configuration is fixed: 256 logical experts, 8 expert groups, 288 physical slots, 4 nodes, 32 GPUs, rebalance every 100 steps, scoring on the next window. A method must accept these as arguments and must not change them.

## Interface and Measurement

- A method is one self-contained policy file `outputs/eplb/<name>.py` (at most 64 KB) defining `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus) -> (phy2log, log2phy, logcnt)`.
- It may import only `torch` and the computational standard library. It must be deterministic.
- It must not use threads, processes, files, sockets, native code, TorchScript or `torch.compile`, and must not interfere with the evaluator's work measurement.
- Score with `scripts/evaluate_train.sh` / `scripts/evaluate.sh` unchanged. Do not edit `eval/`, the anchor policy, or the data files to change scores.

## Compliance Checklist

- [ ] Policy files are self-contained and deterministic
- [ ] Placements are valid for every window (budget, coverage, consistent maps, groups whole on equally loaded nodes)
- [ ] Balancedness is not below the frontier reference by more than 0.005
- [ ] No use of held-out layers, the raw trace, or evaluator internals
- [ ] No wall-clock timing replaces the evaluator's work measure
