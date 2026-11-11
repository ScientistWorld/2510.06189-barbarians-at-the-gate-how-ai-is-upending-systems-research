# Overview

- **Paper ID:** 2510.06189
- **Title:** Barbarians at the Gate: How AI is Upending Systems Research
- **Domain:** Systems / AI-driven algorithm discovery
- **TL;DR:** LLM-driven evolutionary coding (ADRS) can discover strong systems algorithms when candidates are scored by reliable simulators.

## Short Summary

The paper proposes AI-Driven Research for Systems (ADRS), a generate-evaluate-refine loop for systems performance problems, and presents several case studies run with OpenEvolve. This gym packages the MoE Expert Parallelism Load Balancing (EPLB) case study on the paper's own released expert-load trace (DeepSeek-style 58-layer, 256-expert model; 288 replicas, 8 groups, 4 nodes, 32 GPUs). Each rebalance is judged on the next 100-step window, exactly as in the paper's evaluator.

## Key Results (paper)

- Greedy initial program (public DeepSeek/vLLM EPLB): ~540 ms per rebalance, imbalance factor 0.66.
- Frontier-lab reference (non-public, tensorised zigzag): 19.6 ms, same imbalance factor.
- Evolved policy: 3.7 ms (5.3x faster than the reference), same imbalance factor.

## Reproduced Here

The evaluator runs each policy itself and measures its rebalance work deterministically (no wall-clock). The reproduced evolved policy keeps the reference's next-window balance and needs several times less work than the frontier reference. The greedy initial program needs tens of times more. See `HARDENING.md` for the measured numbers.

---

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

---

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

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 32m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 15:49] - none
- Audited inherited workspace and found only template docs/scripts with empty `method/`, `eval/`, and `baseline/`; no prior reproduction was present.
- Implemented a focused MoE Expert Parallelism Load Balancer reproduction with a deterministic trace generator, DeepSeek-style greedy baseline, evolved overloaded-replication plus zigzag placement method, and independent evaluator.
- Local smoke test passed: `bash scripts/reproduce.sh && python validate.py --compare` produced valid scores for 42 traces. A CPU test job is being submitted before raising the milestone.

### [2026-06-02 15:53] - core_claim
- CPU test job `e6526d2b-756` completed successfully in the target container with exit code 0.
- Verified `scripts/run.sh` runs `scripts/reproduce.sh` and `python validate.py --compare` end-to-end.
- Reproduced the core EPLB claim on deterministic traces: the evolved zigzag policy has `work_efficiency_vs_greedy=2.023`, `valid_fraction=1.0`, and `mean_imbalance_factor=0.995`.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 15:54] - method_runs
- Implemented the paper's EPLB case-study algorithm on deterministic expert-load traces.
- Verified that baseline and evolved placement artifacts run end-to-end and produce valid outputs.

### [2026-06-02 16:00] - core_claim
- Reproduced the core EPLB claim: the evolved zigzag policy reduces rebalance work while maintaining valid, balanced GPU placements.
- `scripts/reproduce.sh` and `python validate.py --compare` pass locally.

### [2026-06-02 16:20] - method_runs
- Continuation audit found the earlier `core_claim` was premature: the method used Python snake loops instead of PyTorch tensorized placement, stage-1 zigzag did not affect later stages, and scoring compared stale/partial artifacts.
- Patched the method to use PyTorch tensorized snake placement, added a frontier-reference zigzag baseline, generalized artifact discovery in `eval/`, and prepared a CPU test submission to regenerate scores.

### [2026-06-02 16:32] - core_claim
- CPU test job `934efd9a-f8c` completed with exit code 0.
- Corrected tensorized EPLB reproduction passed `python validate.py --compare`.
- `evolved_zigzag` reached `work_efficiency_vs_frontier_reference=6.711`, `mean_imbalance_factor=0.989`, and `valid_fraction=1.0`, supporting the core claim.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

### [2026-06-02 17:58] - core_claim
- Final verifier CPU job `371d3b15-229` completed successfully after the evaluator hardening pass.
- Hardened scoring now penalizes missing trace outputs, emits completeness and robustness metrics, and still supports the core EPLB claim.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

### [2026-06-02 17:58] - core_claim
- Final verifier CPU job `371d3b15-229` completed successfully after the evaluator hardening pass.
- Hardened scoring now penalizes missing trace outputs, emits completeness and robustness metrics, and still supports the core EPLB claim.

### [2026-06-02 18:05] - core_claim
- Wired the method, baseline, and reproduction scripts to regenerate `scores.json`, `scores_train.json`, and `scores_test.json` from fresh artifacts.
- Prepared a CPU container verification job for the train/test split; `core_claim_plus` will only be claimed after that job completes successfully.

### [2026-06-02 18:20] - core_claim
- Final verification passed `python validate.py --compare`; eval/method separation, shared-path checks, and train/test wiring are valid.
- Sanitized method-agnostic evaluation documentation so from-scratch agents see the task and scoring contract without solution details.
- Kept milestone at `core_claim` because the core EPLB claim is supported on deterministic synthetic traces, with the exact paper traces and non-public implementation still unavailable.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

### [2026-06-02 17:58] - core_claim
- Final verifier CPU job `371d3b15-229` completed successfully after the evaluator hardening pass.
- Hardened scoring now penalizes missing trace outputs, emits completeness and robustness metrics, and still supports the core EPLB claim.

### [2026-06-02 18:05] - core_claim
- Wired the method, baseline, and reproduction scripts to regenerate `scores.json`, `scores_train.json`, and `scores_test.json` from fresh artifacts.
- Prepared a CPU container verification job for the train/test split; `core_claim_plus` will only be claimed after that job completes successfully.

### [2026-06-02 18:20] - core_claim
- Final verification passed `python validate.py --compare`; eval/method separation, shared-path checks, and train/test wiring are valid.
- Sanitized method-agnostic evaluation documentation so from-scratch agents see the task and scoring contract without solution details.
- Kept milestone at `core_claim` because the core EPLB claim is supported on deterministic synthetic traces, with the exact paper traces and non-public implementation still unavailable.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 52m | **GPU**: 0.0h
- **Jobs**: 8 total (7 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.

### [2026-06-02 16:55] - core_claim
- Continuation audit confirmed the implementation is not a surrogate: it directly implements overloaded-expert replication and tensorized snake/zigzag placement from the paper's EPLB case study.
- Corrected the scoring contract so paper-reference targets compare the evolved policy against the frontier-reference baseline, while greedy operation-count speedup remains a diagnostic metric.

### [2026-06-02 17:09] - core_claim
- CPU container verification job `a288216f-231` completed successfully.
- `scripts/reproduce.sh` regenerated EPLB outputs and `python validate.py --compare` passed all scoring and structure checks.

### [2026-06-02 17:58] - core_claim
- Final verifier CPU job `371d3b15-229` completed successfully after the evaluator hardening pass.
- Hardened scoring now penalizes missing trace outputs, emits completeness and robustness metrics, and still supports the core EPLB claim.

### [2026-06-02 18:05] - core_claim
- Wired the method, baseline, and reproduction scripts to regenerate `scores.json`, `scores_train.json`, and `scores_test.json` from fresh artifacts.
- Prepared a CPU container verification job for the train/test split; `core_claim_plus` will only be claimed after that job completes successfully.

### [2026-06-02 18:20] - core_claim
- Final verification passed `python validate.py --compare`; eval/method separation, shared-path checks, and train/test wiring are valid.
- Sanitized method-agnostic evaluation documentation so from-scratch agents see the task and scoring contract without solution details.
- Kept milestone at `core_claim` because the core EPLB claim is supported on deterministic synthetic traces, with the exact paper traces and non-public implementation still unavailable.

### [2026-11-11] - core_claim (hardening, iteration 11)
- Rebuilt the gym on the paper's released expert-load trace and evaluator protocol (288 replicas, 8 groups, 4 nodes, 32 GPUs, 100-step windows, scored on the next window); greedy and frontier reach the paper's 0.66 balance.
- Methods are policy files run by the evaluator; the hidden test streams held-out, relabelled layers one window at a time and measures rebalance work itself (calibrated to CPU time, deterministic).
- Reference `evolved_zigzag`: 2.79x less work than the frozen frontier reference with identical balance (0.6583 test / 0.6674 train); honest leaner variants score higher (3.87x, 4.14x); weaker, simple and brute-force variants and all audited shortcuts score lower or tie. Details in `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: core_claim**
