#!/usr/bin/env python3
"""Create deterministic MoE expert-load traces for EPLB evaluation.

The paper evaluates load changes induced by ShareGPT and GSM8K. Those exact
traces are not released with the paper, so this generator creates a compact
substitute with the same relevant structure: heavy-tailed expert popularity,
bursty hot experts, and distribution shifts across trace families.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from eplb_common import write_json


def make_loads(
    rng: random.Random,
    num_experts: int,
    hot_count: int,
    concentration: float,
    noise: float,
) -> list[int]:
    base = [1.0 / ((i + 1) ** concentration) for i in range(num_experts)]
    rng.shuffle(base)
    hot = rng.sample(range(num_experts), hot_count)
    for rank, expert_id in enumerate(hot):
        base[expert_id] *= 2.8 - 0.25 * rank
    total = sum(base)
    scale = rng.randint(18_000, 32_000) / total
    loads = []
    for value in base:
        jitter = rng.uniform(1.0 - noise, 1.0 + noise)
        loads.append(max(1, int(round(value * scale * jitter))))
    return loads


def main() -> None:
    rng = random.Random(251006189)
    traces = []
    families = [
        ("sharegpt_like", 1.08, 0.18, 7),
        ("gsm8k_like", 0.82, 0.11, 5),
        ("shifted_dialog", 1.22, 0.24, 9),
    ]
    trace_id = 0
    for family, concentration, noise, hot_count in families:
        for _ in range(14):
            num_experts = 64
            num_groups = 8
            num_nodes = 2
            num_gpus = 8
            num_replicas = 96
            phase = 1.0 + 0.15 * math.sin(trace_id / 3.0)
            loads = make_loads(
                rng,
                num_experts=num_experts,
                hot_count=hot_count,
                concentration=concentration * phase,
                noise=noise,
            )
            traces.append(
                {
                    "id": f"{family}_{trace_id:03d}",
                    "family": family,
                    "loads": loads,
                    "num_groups": num_groups,
                    "num_nodes": num_nodes,
                    "num_gpus": num_gpus,
                    "num_replicas": num_replicas,
                }
            )
            trace_id += 1

    payload = {
        "description": "Deterministic EPLB load traces with heavy-tailed and shifting expert popularity.",
        "traces": traces,
    }
    write_json(Path(__file__).with_name("eplb_traces.json"), payload)


if __name__ == "__main__":
    main()
