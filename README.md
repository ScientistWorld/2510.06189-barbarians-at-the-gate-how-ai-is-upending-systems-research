# Overview

- **Paper ID:** 2510.06189
- **Title:** Barbarians at the Gate: How AI is Upending Systems Research
- **Domain:** Systems / AI-driven algorithm discovery
- **TL;DR:** The paper shows that LLM-driven evolutionary coding can discover strong systems algorithms when candidates are scored by reliable simulators.

## Short Summary

The paper proposes AI-Driven Research for Systems (ADRS), a generate-evaluate-refine loop for systems performance problems. It studies several case studies using OpenEvolve, including deadline scheduling, MoE expert placement, LLM-SQL layout optimization, and transaction scheduling. This reproduction packages the MoE expert-placement case study because it has a compact public baseline and a clearly described evolved heuristic.

## Key Results

- Paper reports the evolved EPLB algorithm matching baseline imbalance factor while reducing rebalance runtime from 19.6 ms to 3.7 ms versus a non-public reference.
- The initial public DeepSeek-style greedy EPLB policy is reported at about 540 ms with imbalance factor 0.66.
- The reproduced environment scores the same trade-off with valid placement, imbalance, and machine-independent operation counts.

---

# Problem Definition

## Research Question

Can automated algorithm discovery improve small, performance-critical systems algorithms when a reliable simulator can score correctness and performance? This reproduction focuses on the MoE expert-parallel load-balancing task: given changing per-expert token loads, a fixed number of logical experts, physical expert replicas, nodes, and GPUs, produce a replica count and GPU placement that balances generated-token load across GPUs.

## Why It Matters

Mixture-of-Experts inference can become bottlenecked when popular experts concentrate on a subset of GPUs. A good load balancer lets serving systems adapt to workload shifts while keeping reconfiguration overhead small.

## Success Criteria

A successful solution produces valid placements for all traces, maintains high average/max GPU load balance, and reduces the amount of work needed to rebalance experts when loads change.

---

# Paper's Method

## Key Contribution

The paper argues for AI-Driven Research for Systems (ADRS): use LLM-driven iterative code generation plus reliable simulators to discover systems algorithms. The reproduced case study is Expert Parallelism Load Balancing (EPLB) for MoE inference.

## Approach

For EPLB, the algorithmic pipeline has three stages: distribute expert groups across nodes, create replicas for overloaded experts, and assign physical replicas to GPUs. The initial program uses greedy bin packing. The evolved policy keeps the overloaded-expert replication idea but replaces loop-heavy placement with a PyTorch tensorized snake/zigzag assignment that sorts replicas and alternates pack direction across blocks. This reproduction makes the group-to-node zigzag order feed into physical replica materialization, then applies the same tensorized placement at the GPU stage. The paper emphasizes that applying this zigzag partitioning across multiple EPLB stages was the key improvement.

## Main Claims

- Primary: ADRS can discover a faster EPLB rebalancing algorithm without worsening load balance.
- Reproduced target: the evolved zigzag placement reduces rebalance work relative to greedy while keeping valid, similarly balanced placements.
- Secondary paper-level claim: reliable evaluators make systems algorithm discovery especially suitable for AI-driven search.

---

# Evaluation

## Metrics

- `mean_imbalance_factor`: average over traces of `(mean GPU load) / (max GPU load)`. Higher is better; 1.0 is perfectly balanced.
- `work_efficiency_vs_greedy`: greedy baseline mean operation count divided by candidate mean operation count. Higher is better. This replaces wall-clock runtime with a machine-independent work proxy.
- `work_efficiency_vs_frontier_reference`: tensorized frontier-reference baseline mean operation count divided by candidate mean operation count. Higher is better and is the primary metric because the paper's headline EPLB result compares against that reference implementation.
- `mean_operation_count`: counted algorithmic operations for group packing, replication, and placement. Lower is better.
- `valid_fraction`: fraction of traces whose placement has the requested number of replicas, valid expert ids, and positive replica count for every logical expert. Higher is better.

## Baselines and Targets

The baselines are the public DeepSeek EPLB-style greedy bin-packing policy and a frontier-reference-style tensorized zigzag baseline. The paper reports an initial greedy runtime of about 540 ms with imbalance factor 0.66, a non-public reference runtime of 19.6 ms with the same imbalance, and an evolved runtime of 3.7 ms with the same imbalance. This environment keeps those wall-clock values as `reported_runtime_ms`, but the scored speed metrics use operation-count ratios from the reproducible benchmark because wall-clock runtime is hardware-dependent.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to generate deterministic expert-load traces, run the baseline and candidate method, and write `scoring/scores.json`. Standalone evaluation is `bash scripts/evaluate.sh`; it reads placement artifacts from `outputs/eplb/` and does not import from `method/`.

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


---

# Reproduction Milestones

**Current: core_claim**
