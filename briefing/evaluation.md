# Evaluation

## Data

The workload is the expert-load trace released with the paper's EPLB case study (Hugging Face `abmfy/eplb-openevolve`, `expert-load.json`): 1174 recorded steps of a 58-layer, 256-expert MoE model serving ShareGPT and GSM8K requests. Consecutive steps are summed into 100-step windows, as in the paper's evaluator. Windows start at four offsets (0, 25, 50, 75 steps), giving four sequences of 11-12 windows and 43 rebalances per method.

- Visible (train): the 29 even MoE layers, in `data/eplb_train_windows.json` (expert ids relabelled within groups; `scripts/download.sh` rebuilds it).
- Hidden (test): the 29 held-out layers, relabelled with a secret permutation, used only by the hidden test. Do not fetch the raw trace to read them.

## Submission Interface

A method is one self-contained Python file `outputs/eplb/<method_name>.py` defining the paper's (vLLM's) interface:

```python
rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
    -> (phy2log, log2phy, logcnt)
```

- `weight`: int64 tensor `[num_layers, 256]`, the load of the window that just ended.
- `phy2log`: `[num_layers, 288]`, logical expert held by each physical slot (slot `p` is on GPU `p // 9`, node `p // 72`).
- `log2phy`: `[num_layers, 256, X]`, the slots of each expert in its first `logcnt` entries, `-1` after that (any `X` >= the largest replica count).
- `logcnt`: `[num_layers, 256]`, replicas per expert.

Every policy file in `outputs/eplb/` is evaluated; the method name is the file name. The evaluator calls the policy once per window, in order, on a fresh module for each sequence. Module-level state may carry across the calls of one sequence.

Policy rules:
- Self-contained, at most 64 KB. It may import only `torch` and the computational standard library (no numpy, threads, processes, files, sockets, ctypes, `sys`, `os`, `gc`, `importlib`, ...).
- Deterministic: the same inputs give the same outputs.
- No TorchScript/`torch.compile`, no native extensions, and no tampering with the evaluator or its measurement.

Violations are detected and fail the run.

## Validity

Each returned placement must satisfy all of the following, or the whole run fails:
- tensors of the exact shapes above, integer dtype;
- every logical expert has at least one replica; replicas fill exactly 288 slots; `logcnt` matches `phy2log`;
- `log2phy` lists exactly each expert's slots, padded with `-1`;
- each expert group sits entirely on one node, and every node hosts exactly 2 groups.

## Metrics

Scored (equal coefficients):
- `speedup_vs_frontier_reference` (primary, higher is better): the frozen frontier-lab reference policy's mean rebalance work divided by the method's mean rebalance work, measured on the same windows.
- `balancedness` (higher is better): the paper's imbalance factor on the next window, `sum over layers of mean GPU load / sum over layers of max GPU load`, averaged over all rebalances.

Constraint: a method whose balancedness is more than 0.005 below the frontier reference on the same windows fails the run.

Unscored diagnostics: `mean_rebalance_cost`, `balance_change_vs_frontier_reference`, `worst_window_balancedness`, `valid_fraction`, `run_failed`, `num_rebalances`, and `diagnostic_cpu_ms_per_rebalance` (host CPU time of an unmetered replay, only a cross-check).

## Rebalance Work Measure

The evaluator runs the policy in a sandboxed process and counts the work of each `rebalance_experts` call itself. Self-reported numbers are ignored. One work unit is roughly one interpreted Python bytecode (about 14 ns on the calibration host):

| event | units |
|---|---|
| Python bytecode instruction executed (any code run by the call) | 1 |
| call from Python into a C function | 1 |
| element of a Python container scanned by such a call (`sorted`, `sum`, `max`, `list(...)`, `torch.tensor(list)`, `.tolist()`, `list.index`, ...; sorts pay `n log2 n`, heap/bisect ops `log2 n`) | 2 |
| PyTorch kernel dispatched | 20 |
| tensor element read or written by a non-view kernel (sort-like kernels times `log2` of the sorted length) | 0.06 |

The weights were calibrated once against unmetered CPU time of greedy-loop, tensorised, pure-Python and search-heavy policies on the job host. Work units track CPU time within about 2x across these very different implementation styles, and they rank those policies in the same order as CPU time.

Work done at import time is not counted, but it cannot depend on the windows. A single rebalance above 5e8 units fails the run. Wall-clock time is never scored.

## Failure Handling

Any failure gives the method the worst value in every scored metric (0 and 0), with the reason under `_failures` in the score file. Failures include: a load error, an exception, an invalid placement, a guard violation, non-determinism, exceeding the work cap, or the balance constraint. Missing or malformed files never leave stale scores behind.

## Commands

- `bash scripts/evaluate_train.sh` (or `scripts/evaluate.sh`): score every policy in `outputs/eplb/` on the visible windows, written to `scoring/scores_train.json` (or `scores.json`).
- `bash scripts/evaluate_test.sh`: the hidden test, with the same interface on held-out windows, written to `scoring/scores_test.json`.
- `bash scripts/reproduce.sh`: publish the shipped baselines and method, then run both evaluations.
