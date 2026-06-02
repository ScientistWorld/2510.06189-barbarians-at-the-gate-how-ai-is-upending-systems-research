# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 15:49] - none
- Audited inherited workspace and found only template docs/scripts with empty `method/`, `eval/`, and `baseline/`; no prior reproduction was present.
- Implemented a focused MoE Expert Parallelism Load Balancer reproduction with a deterministic trace generator, DeepSeek-style greedy baseline, evolved overloaded-replication plus zigzag placement method, and independent evaluator.
- Local smoke test passed: `bash scripts/reproduce.sh && python validate.py --compare` produced valid scores for 42 traces. A CPU test job is being submitted before raising the milestone.

### [2026-06-02 15:53] - core_claim
- CPU test job `e6526d2b-756` completed successfully in the target container with exit code 0.
- Verified `scripts/run.sh` runs `scripts/reproduce.sh` and `python validate.py --compare` end-to-end.
- Reproduced the core EPLB claim on deterministic traces: the evolved zigzag policy has `work_efficiency_vs_greedy=2.023`, `valid_fraction=1.0`, and `mean_imbalance_factor=0.995`.

## Stop Justification

Reached `core_claim`: the packaged environment implements the paper-described EPLB computational pipeline, scores both benefit and constraint metrics, and passed a compute-container smoke test. Remaining gaps are documented deviations: unavailable original ShareGPT/GSM8K traces, unavailable non-public frontier reference implementation, and machine-independent operation counts replacing wall-clock runtime.
