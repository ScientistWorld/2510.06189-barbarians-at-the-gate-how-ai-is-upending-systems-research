# Barbarians at the Gate: How AI is Upending Systems Research

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2510.06189"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2510.06189-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2510.06189](https://arxiv.org/abs/2510.06189) |
| **Domain** | Systems / AI-driven algorithm discovery |
| **TL;DR** | The paper shows that LLM-driven evolutionary coding can discover strong systems algorithms when candidates are scored by reliable simulators. |

## Summary

The paper proposes AI-Driven Research for Systems (ADRS), a generate-evaluate-refine loop for systems performance problems. It studies several case studies using OpenEvolve, including deadline scheduling, MoE expert placement, LLM-SQL layout optimization, and transaction scheduling. This reproduction packages the MoE expert-placement case study because it has a compact public baseline and a clearly described evolved heuristic.

## Problem

Can automated algorithm discovery improve small, performance-critical systems algorithms when a reliable simulator can score correctness and performance? This reproduction focuses on the MoE expert-parallel load-balancing task: given changing per-expert token loads, a fixed number of logical experts, physical expert replicas, nodes, and GPUs, produce a replica count and GPU placement that balances generated-token load across GPUs.

## The paper's method

The paper argues for AI-Driven Research for Systems (ADRS): use LLM-driven iterative code generation plus reliable simulators to discover systems algorithms. The reproduced case study is Expert Parallelism Load Balancing (EPLB) for MoE inference.

## Key results of the paper

- Paper reports the evolved EPLB algorithm matching baseline imbalance factor while reducing rebalance runtime from 19.6 ms to 3.7 ms versus a non-public reference.
- The initial public DeepSeek-style greedy EPLB policy is reported at about 540 ms with imbalance factor 0.66.
- The reproduced environment scores the same trade-off with valid placement, imbalance, and machine-independent operation counts.

## Run it

```bash
pip install sciworld
sw research 2510.06189 --mode innovate --agent claude-code
sw evaluate 2510.06189
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

