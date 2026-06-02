# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-02 15:54] - method_runs
- Implemented the paper's EPLB case-study algorithm on deterministic expert-load traces.
- Verified that baseline and evolved placement artifacts run end-to-end and produce valid outputs.

### [2026-06-02 16:00] - core_claim
- Reproduced the core EPLB claim: the evolved zigzag policy reduces rebalance work while maintaining valid, balanced GPU placements.
- `scripts/reproduce.sh` and `python validate.py --compare` pass locally.

### [2026-06-02 16:20] - method_runs
- Continuation audit found the earlier `core_claim` was premature: the method used Python snake loops instead of PyTorch tensorized placement, stage-1 zigzag did not affect later stages, and scoring compared stale/partial artifacts.
- Patched the method to use PyTorch tensorized snake placement, added a frontier-reference zigzag baseline, generalized artifact discovery in `eval/`, and prepared a CPU test submission to regenerate scores.

## Stop Justification
- Not stopped. Current submission is validating the corrected tensorized implementation.
