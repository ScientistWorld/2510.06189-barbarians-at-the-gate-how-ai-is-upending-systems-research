#!/usr/bin/env python3
"""Frontier-reference-style EPLB baseline from the paper.

The paper's non-public reference replaces the final greedy GPU bin-packing loop
with a tensorized zigzag/snake layout. This reimplementation keeps the public
DeepSeek-style group distribution and overloaded-expert replication, then uses
the tensor snake only for physical-replica placement. The evolved method in
`method/` reuses the same zigzag idea across more EPLB stages.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "baseline"))

from eplb_baseline import greedy_pack, replicate_experts
from eplb_common import load_traces, write_json


def tensor_snake_order(weights: torch.Tensor, num_packs: int, capacity: int) -> tuple[list[int], int]:
    if weights.numel() != num_packs * capacity:
        raise ValueError("snake packing expects exactly num_packs * capacity items")
    sorted_idx = torch.argsort(weights, descending=True, stable=True)
    positions = torch.arange(weights.numel(), device=weights.device)
    block_id = positions // num_packs
    idx_in_block = positions % num_packs
    pack_for_sorted = torch.where(
        block_id.remainder(2).eq(0),
        idx_in_block,
        num_packs - 1 - idx_in_block,
    )
    pack_matrix = torch.empty((num_packs, capacity), dtype=torch.long, device=weights.device)
    pack_matrix[pack_for_sorted, block_id] = sorted_idx
    return [int(x) for x in pack_matrix.reshape(-1).tolist()], 8


def run_trace(trace: dict) -> dict:
    start = time.perf_counter()
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
    physical_to_logical: list[int] = []
    replica_loads: list[float] = []
    for expert_id, count in enumerate(replica_counts):
        physical_to_logical.extend([expert_id] * count)
        replica_loads.extend([loads[expert_id] / count] * count)

    order, placement_ops = tensor_snake_order(
        torch.tensor(replica_loads, dtype=torch.float32),
        num_gpus,
        num_replicas // num_gpus,
    )
    packed = [physical_to_logical[idx] for idx in order]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "trace_id": trace["id"],
        "replica_counts": replica_counts,
        "physical_to_logical": packed,
        "operation_count": group_ops + replica_ops + placement_ops,
        "rebalance_time_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/eplb_traces.json")
    parser.add_argument("--output", default="outputs/eplb/frontier_reference_zigzag.json")
    args = parser.parse_args()

    traces = load_traces(args.traces)["traces"]
    outputs = [run_trace(trace) for trace in traces]
    write_json(args.output, {"method": "frontier_reference_zigzag", "outputs": outputs})


if __name__ == "__main__":
    main()
