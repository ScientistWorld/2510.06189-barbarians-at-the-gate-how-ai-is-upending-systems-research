#!/usr/bin/env python3
"""DeepSeek-style greedy EPLB baseline.

This mirrors the public EPLB structure used as the paper's initial program:
1. greedily pack expert groups to nodes,
2. replicate hot experts by repeatedly splitting the largest effective load,
3. greedily pack physical replicas to GPUs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from eplb_common import expert_replica_loads, load_traces, write_json


def greedy_pack(weights: list[float], num_packs: int, capacity: int) -> tuple[list[int], int]:
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    pack_loads = [0.0] * num_packs
    pack_counts = [0] * num_packs
    assignment = [-1] * len(weights)
    ops = 0
    for item in order:
        best_pack = None
        best_load = None
        for pack_id in range(num_packs):
            ops += 1
            if pack_counts[pack_id] >= capacity:
                continue
            if best_load is None or pack_loads[pack_id] < best_load:
                best_pack = pack_id
                best_load = pack_loads[pack_id]
        if best_pack is None:
            raise ValueError("no pack has capacity left")
        assignment[item] = best_pack
        pack_loads[best_pack] += weights[item]
        pack_counts[best_pack] += 1
    return assignment, ops


def replicate_experts(loads: list[float], num_replicas: int) -> tuple[list[int], int]:
    counts = [1] * len(loads)
    ops = 0
    for _ in range(num_replicas - len(loads)):
        best = 0
        best_effective = loads[0] / counts[0]
        for expert_id in range(1, len(loads)):
            ops += 1
            effective = loads[expert_id] / counts[expert_id]
            if effective > best_effective:
                best = expert_id
                best_effective = effective
        counts[best] += 1
    return counts, ops


def run_trace(trace: dict) -> dict:
    loads = [float(x) for x in trace["loads"]]
    num_groups = int(trace["num_groups"])
    num_nodes = int(trace["num_nodes"])
    num_gpus = int(trace["num_gpus"])
    num_replicas = int(trace["num_replicas"])
    group_size = len(loads) // num_groups

    group_loads = [
        sum(loads[group_id * group_size : (group_id + 1) * group_size])
        for group_id in range(num_groups)
    ]
    _, group_ops = greedy_pack(group_loads, num_nodes, num_groups // num_nodes)

    replica_counts, replica_ops = replicate_experts(loads, num_replicas)
    physical_to_logical = []
    for expert_id, count in enumerate(replica_counts):
        physical_to_logical.extend([expert_id] * count)
    replica_loads = expert_replica_loads(loads, replica_counts)
    gpu_assignment, placement_ops = greedy_pack(
        replica_loads, num_gpus, num_replicas // num_gpus
    )

    packed = [None] * num_replicas
    gpu_slot = [0] * num_gpus
    for physical_id, gpu_id in enumerate(gpu_assignment):
        out_index = gpu_id * (num_replicas // num_gpus) + gpu_slot[gpu_id]
        packed[out_index] = physical_to_logical[physical_id]
        gpu_slot[gpu_id] += 1

    return {
        "trace_id": trace["id"],
        "replica_counts": replica_counts,
        "physical_to_logical": packed,
        "operation_count": group_ops + replica_ops + placement_ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/eplb_traces.json")
    parser.add_argument("--output", default="outputs/eplb/greedy_deepseek.json")
    args = parser.parse_args()

    traces = load_traces(args.traces)["traces"]
    outputs = [run_trace(trace) for trace in traces]
    write_json(args.output, {"method": "greedy_deepseek", "outputs": outputs})


if __name__ == "__main__":
    main()
