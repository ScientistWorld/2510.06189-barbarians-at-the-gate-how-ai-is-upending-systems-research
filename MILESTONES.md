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

### [2026-11-11] - core_claim (hardening, iteration 11)
- Rebuilt the gym on the paper's released expert-load trace and evaluator protocol (288 replicas, 8 groups, 4 nodes, 32 GPUs, 100-step windows, scored on the next window); greedy and frontier reach the paper's 0.66 balance.
- Methods are policy files run by the evaluator; the hidden test streams held-out, relabelled layers one window at a time and measures rebalance work itself (calibrated to CPU time, deterministic).
- Reference `evolved_zigzag`: 2.79x less work than the frozen frontier reference with identical balance (0.6583 test / 0.6674 train); honest leaner variants score higher (3.87x, 4.14x); weaker, simple and brute-force variants and all audited shortcuts score lower or tie. Details in `HARDENING.md`.

### [2026-11-17] - core_claim (hardening, iteration 12)
- Rebalance work is now counted outside the policy process: retired user-space CPU instructions from a hardware counter the evaluator attaches to the policy process tree. Each pair runs in a freshly forked, Landlock-sandboxed process (no workspace access, no exec, no writes), one scored call per process.
- Hidden test moved to held-out time blocks (odd layers × time blocks 1 and 3; visible: even layers × blocks 0 and 2), 20 pairs per slice.
- Reference `evolved_zigzag`: 4.66x fewer instructions than the frozen frontier reference with identical balance (0.6863 test / 0.6810 train; paper 5.3x, same 0.66 level). Honest leaner variants win on 20/20 paired windows (+1.8%, +8.7%); weaker variants fail the balance gate; every audited shortcut fails, ties or loses. Details in `HARDENING.md`.
