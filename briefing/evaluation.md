# Evaluation

## Data

The workload is the expert-load trace released with the paper's EPLB case study (Hugging Face `abmfy/eplb-openevolve`, `expert-load.json`): 1174 recorded steps of a 58-layer, 256-expert MoE model serving ShareGPT and GSM8K requests. Consecutive steps are summed into 100-step windows, as in the paper's evaluator. A rebalance pair is an input window (steps `[s, s+100)`) and the window that follows it (`[s+100, s+200)`), on which the placement is scored.

The trace is cut into four equal time blocks. Each slice holds 20 pairs, 10 per block, starting every 10 steps:

- Visible (train): the 29 even MoE layers in time blocks 0 and 2 (steps 0-294 and 587-880), in `data/eplb_train_windows.json`. Expert ids are relabelled within groups. `scripts/download.sh` rebuilds the file.
- Hidden (test): the 29 odd MoE layers in time blocks 1 and 3 (steps 294-587 and 880-1174), relabelled with a secret permutation, used only by the hidden test. No visible layer covers a hidden time block, and no hidden layer is visible at any time. Do not fetch the raw trace to read them.

## Submission Interface

A method is one self-contained Python file `outputs/eplb/<method_name>.py` defining the paper's (vLLM's) interface:

```python
rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
    -> (phy2log, log2phy, logcnt)
```

- `weight`: int64 tensor `[num_layers, 256]`, the load of the window that just ended.
- `phy2log`: `[num_layers, 288]`, logical expert held by each physical slot (slot `p` is on GPU `p // 9`, node `p // 72`).
- `log2phy`: `[num_layers, 256, X]`, the slots of each expert in its first `logcnt` entries, `-1` after that (any `X` with largest replica count <= `X` <= 288).
- `logcnt`: `[num_layers, 256]`, replicas per expert.

All outputs are plain `torch.Tensor`s with an integer dtype. Every policy file in `outputs/eplb/` is evaluated; the method name is the file name.

How a policy is run, for every pair:
1. a fresh policy process is started and the policy file is executed (import);
2. one warm-up call runs on a synthetic window (unscored, so one-time library initialisation is not charged to the scored call);
3. the policy file is executed again in a fresh module;
4. one scored call receives the pair's input window;
5. the process and everything it started are then ended.

Nothing kept in memory carries over between pairs, so a policy sees exactly one real window per process.

The policy process is sandboxed by the operating system (Linux Landlock):
- read-only access to the Python, torch and numpy installation;
- no access to the workspace or any other file;
- no file writes, no starting programs, no network;
- no signals to processes outside the sandbox.

A policy may import `torch`, `numpy` and the standard library, and may use threads or forked processes, but all of that work is counted (see below). Policy files are limited to 64 KB.

## Validity

Each returned placement must satisfy all of the following, or the whole run fails:
- tensors of the exact shapes above, integer dtype;
- every logical expert has at least one replica; replicas fill exactly 288 slots; `logcnt` matches `phy2log`;
- `log2phy` lists exactly each expert's slots, padded with `-1`;
- each expert group sits entirely on one node, and every node hosts exactly 2 groups.

## Metrics

Scored (equal coefficients):
- `speedup_vs_frontier_reference` (primary, higher is better): the frozen frontier-lab reference policy's mean rebalance work divided by the method's mean rebalance work, both measured on the same pairs in the same run.
- `balancedness` (higher is better): the paper's imbalance factor on the next window, `sum over layers of mean GPU load / sum over layers of max GPU load`, averaged over all pairs.

Constraint: a method whose balancedness is more than 0.005 below the frontier reference on the same pairs fails the run.

Unscored diagnostics: `mean_rebalance_instructions`, `balance_change_vs_frontier_reference`, `worst_window_balancedness`, `valid_fraction`, `run_failed`, `num_rebalances`, and `diagnostic_cpu_ms_per_rebalance` (host CPU time of the measured call, a cross-check only).

## Rebalance Work Measure

Rebalance work is the number of CPU instructions the policy process executes for the scored call. The evaluator counts them from outside the policy process with the processor's hardware counter of retired user-space instructions. The counter is attached before the policy is loaded and also covers every thread and process the policy starts. The count runs from the moment the input window is sent until the complete placement has been received, so it includes turning the returned tensors into bytes.

- Nothing in the policy process takes part in the measurement, and self-reported numbers are ignored.
- Every operation is charged by the instructions it actually executes, in Python, torch, numpy or any other library.
- Kernel work (page faults, system calls) is not counted.
- The count does not depend on machine load. Repeated runs agree to about 0.05%, and two copies of the same code under different file names to about 0.3%.
- The torch kernels are pinned to the same CPU code path on every x86 host (`ATEN_CPU_CAPABILITY=default`).
- Instructions are not time: code with many cheap instructions (a Python interpreter loop) retires them faster than code dominated by library dispatch. The measure rewards executing fewer instructions.

Work done at import is not counted, but it cannot depend on the windows. A single call above 1e11 instructions (roughly 30 s of CPU) fails the run, as does a process that makes no progress at all for 10 minutes. No wall-clock limit applies to a policy that is computing.

## Failure Handling

Any failure gives the method the worst value in every scored metric (0 and 0), with the reason under `_failures` in the score file. Failures include: a load error, an exception, an invalid output or placement, exceeding the instruction cap, a stalled process, or the balance constraint. Every method is recorded as failed before evaluation starts, so an interrupted evaluation never leaves stale or missing scores.

## Commands

- `bash scripts/evaluate_train.sh` (or `scripts/evaluate.sh`): score every policy in `outputs/eplb/` on the visible pairs, written to `scoring/scores_train.json` (or `scores.json`).
- `bash scripts/evaluate_test.sh`: the hidden test, with the same interface on the held-out pairs, written to `scoring/scores_test.json`.
- `bash scripts/reproduce.sh`: publish the shipped baselines and method, then run both evaluations.

Evaluation needs Linux with `perf_event_open` available to the user (`/proc/sys/kernel/perf_event_paranoid` at most 2), as on the job nodes; without it the evaluator stops with an explanation instead of scoring.
