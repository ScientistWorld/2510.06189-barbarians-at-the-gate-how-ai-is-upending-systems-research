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
