#!/usr/bin/env python3
"""Paper's evolved EPLB placement heuristic.

The reproduction keeps the public EPLB replication stage but replaces the
loop-heavy final bin packing with the tensorized "snake" placement pattern
described in Figure 5 of the paper: sorted physical replicas are assigned to
GPUs in alternating forward and reversed blocks. The same tensorized pattern is
also used for group-to-node assignment, matching the paper's note that the
breakthrough came from reusing zigzag partitioning across EPLB stages.
"""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from eplb_common import load_traces, write_json


def heap_replicate_overloaded(loads: list[float], num_replicas: int) -> tuple[list[int], int]:
    """Same overloaded-expert split rule as the baseline, with a heap.

    The greedy baseline linearly scans all logical experts each time it adds a
    replica. The paper's evolved code reduces Python-loop overhead while
    preserving the intuitive rule of splitting overloaded experts; a max-heap
    is a dependency-free way to reproduce that computational step.
    """
    counts = [1] * len(loads)
    heap = [(-load, expert_id) for expert_id, load in enumerate(loads)]
    heapq.heapify(heap)
    ops = len(loads)
    for _ in range(num_replicas - len(loads)):
        _, expert_id = heapq.heappop(heap)
        counts[expert_id] += 1
        heapq.heappush(heap, (-(loads[expert_id] / counts[expert_id]), expert_id))
        ops += max(1, len(loads).bit_length())
    return counts, ops


def tensor_snake_pack(
    weights: torch.Tensor,
    num_packs: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tensorized zigzag/snake placement from Figure 5.

    Returns both the ordered item ids, where contiguous slots are a pack, and a
    per-item pack assignment. The implementation follows the paper's key
    computational step: sort once, build block/position tensors, reverse odd
    blocks with `torch.where`, and scatter into a pack-by-slot matrix.
    """
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
    slot_for_sorted = block_id
    pack_matrix = torch.empty((num_packs, capacity), dtype=torch.long, device=weights.device)
    pack_matrix[pack_for_sorted, slot_for_sorted] = sorted_idx
    per_item_pack = torch.empty(weights.numel(), dtype=torch.long, device=weights.device)
    per_item_pack[sorted_idx] = pack_for_sorted

    # Machine-independent work proxy: tensor dispatches for sort, arange,
    # block/position arithmetic, where, and two scatters.
    tensor_ops = 8
    return pack_matrix.reshape(-1), per_item_pack, tensor_ops


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
    group_order_t, group_to_node_t, group_ops = tensor_snake_pack(
        torch.tensor(group_loads, dtype=torch.float32),
        num_nodes,
        num_groups // num_nodes,
    )
    group_order = [int(x) for x in group_order_t.tolist()]

    replica_counts, replica_ops = heap_replicate_overloaded(loads, num_replicas)

    # Stage 1 now affects later stages: physical replicas are materialized in
    # group/node zigzag order before the final GPU-level snake placement.
    logical_order = []
    for group_id in group_order:
        start_expert = group_id * group_size
        logical_order.extend(range(start_expert, start_expert + group_size))

    physical_to_logical: list[int] = []
    replica_loads: list[float] = []
    for expert_id in logical_order:
        count = replica_counts[expert_id]
        physical_to_logical.extend([expert_id] * count)
        replica_loads.extend([loads[expert_id] / count] * count)

    order_t, _, placement_ops = tensor_snake_pack(
        torch.tensor(replica_loads, dtype=torch.float32),
        num_gpus,
        num_replicas // num_gpus,
    )
    order = [int(x) for x in order_t.tolist()]
    packed = [physical_to_logical[idx] for idx in order]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "trace_id": trace["id"],
        "replica_counts": replica_counts,
        "group_to_node": [int(x) for x in group_to_node_t.tolist()],
        "physical_to_logical": packed,
        "operation_count": group_ops + replica_ops + placement_ops,
        "rebalance_time_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/eplb_traces.json")
    parser.add_argument("--output", default="outputs/eplb/evolved_zigzag.json")
    args = parser.parse_args()

    traces = load_traces(args.traces)["traces"]
    outputs = [run_trace(trace) for trace in traces]
    write_json(args.output, {"method": "evolved_zigzag", "outputs": outputs})


if __name__ == "__main__":
    main()
