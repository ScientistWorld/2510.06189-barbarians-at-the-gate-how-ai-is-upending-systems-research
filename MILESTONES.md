# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 16:04] - method_runs
- The EPLB pipeline ran end-to-end on deterministic MoE expert-load traces.
- Generated greedy baseline, frontier-reference zigzag baseline, evolved zigzag artifacts, and reusable scores.

### [2026-06-02 16:30] - core_claim
- Reproduced the paper's EPLB core claim on a scaled deterministic trace substitute.
- The evolved tensorized zigzag policy reduced mean operation count versus the frontier-reference zigzag baseline while preserving valid placements and comparable load balance.
