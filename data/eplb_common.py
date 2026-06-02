"""Shared data structures and metrics for EPLB experiments.

This module is intentionally method-agnostic. It defines the workload format
and how a placement artifact is scored, but it does not implement any candidate
load-balancing policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_traces(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def expert_replica_loads(loads: list[float], replica_counts: list[int]) -> list[float]:
    replica_loads: list[float] = []
    for expert_id, count in enumerate(replica_counts):
        if count <= 0:
            raise ValueError(f"expert {expert_id} has no replica")
        replica_loads.extend([loads[expert_id] / count] * count)
    return replica_loads


def gpu_loads_from_placement(
    loads: list[float],
    replica_counts: list[int],
    physical_to_logical: list[int],
    num_gpus: int,
) -> list[float]:
    if len(physical_to_logical) % num_gpus != 0:
        raise ValueError("physical experts must divide evenly across GPUs")
    expected_replicas = sum(replica_counts)
    if len(physical_to_logical) != expected_replicas:
        raise ValueError(
            f"placement has {len(physical_to_logical)} replicas, expected {expected_replicas}"
        )

    seen = [0] * len(loads)
    gpu_loads = [0.0] * num_gpus
    slots_per_gpu = len(physical_to_logical) // num_gpus
    for physical_id, logical_id in enumerate(physical_to_logical):
        if logical_id < 0 or logical_id >= len(loads):
            raise ValueError(f"invalid logical expert id {logical_id}")
        replica_rank = seen[logical_id]
        seen[logical_id] += 1
        if replica_rank >= replica_counts[logical_id]:
            raise ValueError(f"expert {logical_id} appears more often than its replica count")
        gpu_id = physical_id // slots_per_gpu
        gpu_loads[gpu_id] += loads[logical_id] / replica_counts[logical_id]

    if seen != replica_counts:
        raise ValueError(f"placement replica counts {seen} do not match declared {replica_counts}")
    return gpu_loads


def imbalance_factor(gpu_loads: list[float]) -> float:
    """Return paper's average/max load balance ratio; higher is better."""
    max_load = max(gpu_loads)
    if max_load <= 0:
        return 1.0
    return (sum(gpu_loads) / len(gpu_loads)) / max_load


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
