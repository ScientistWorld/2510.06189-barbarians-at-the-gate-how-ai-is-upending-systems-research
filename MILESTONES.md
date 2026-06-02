# Reproduction Milestones

**Current: core_claim**

## Progress Log

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
