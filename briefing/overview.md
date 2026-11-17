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

The evaluator runs each policy itself in a sandboxed process and counts the CPU instructions of each rebalance with a hardware counter (deterministic, no wall-clock). On the hidden test, the reproduced evolved policy keeps the frontier reference's next-window balance (0.686) with 4.66x fewer instructions (paper: 5.3x less time). The greedy initial program needs about 40x more instructions than the frontier reference. See `HARDENING.md` for the measured numbers.
