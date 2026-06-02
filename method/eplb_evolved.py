#!/usr/bin/env python3
"""Paper's evolved EPLB placement heuristic.

The reproduction keeps the public EPLB replication stage but replaces the
loop-heavy final bin packing with the tensorized "snake" placement pattern
described in Figure 5 of the paper: sorted physical replicas are assigned to
GPUs in alternating forward and reversed blocks. The same pattern is also used
for group-to-node assignment, matching the paper's note that the breakthrough
came from reusing zigzag partitioning across EPLB stages.
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "baseline"))

from eplb_common import expert_replica_loads, load_traces, write_json


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


def snake_pack_order(weights: list[float], num_packs: int, capacity: int) -> tuple[list[int], int]:
    """Return output order where contiguous slots represent packs.

    This is the loop-free pattern from the paper expressed in portable Python:
    after sorting by load, items in even blocks are mapped left-to-right and odd
    blocks right-to-left. The counted operations reflect sort plus vector-style
    index construction, not an item-by-pack linear search.
    """
    if len(weights) != num_packs * capacity:
        raise ValueError("snake packing expects exactly num_packs * capacity items")
    sorted_items = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    packs: list[list[int]] = [[] for _ in range(num_packs)]
    for sorted_pos, item in enumerate(sorted_items):
        block_id = sorted_pos // num_packs
        idx_in_block = sorted_pos % num_packs
        pack_id = idx_in_block if block_id % 2 == 0 else num_packs - 1 - idx_in_block
        packs[pack_id].append(item)
    ordered = []
    for pack in packs:
        if len(pack) != capacity:
            raise ValueError("invalid snake capacity")
        ordered.extend(pack)
    operations = len(weights) + int(len(weights) * max(1, len(weights)).bit_length())
    return ordered, operations


def adaptive_repair(
    ordered: list[int],
    weights: list[float],
    num_packs: int,
    capacity: int,
    max_passes: int = 2,
) -> tuple[list[int], int]:
    """Small balance repair after the paper's zigzag layout.

    This models the paper-described adaptive ordering refinements: keep the
    staggered block layout, then try a bounded number of swaps between the
    heaviest and lightest packs when they reduce max-load imbalance.
    """
    packs = [ordered[i * capacity : (i + 1) * capacity] for i in range(num_packs)]
    loads = [sum(weights[item] for item in pack) for pack in packs]
    ops = 0
    for _ in range(max_passes):
        hi = max(range(num_packs), key=lambda idx: loads[idx])
        lo = min(range(num_packs), key=lambda idx: loads[idx])
        current_gap = loads[hi] - loads[lo]
        if current_gap <= 0:
            break
        best = None
        for hi_pos, hi_item in enumerate(packs[hi]):
            for lo_pos, lo_item in enumerate(packs[lo]):
                ops += 1
                new_hi = loads[hi] - weights[hi_item] + weights[lo_item]
                new_lo = loads[lo] - weights[lo_item] + weights[hi_item]
                new_gap = abs(new_hi - new_lo)
                if new_gap < current_gap and (best is None or new_gap < best[0]):
                    best = (new_gap, hi_pos, lo_pos, new_hi, new_lo)
        if best is None:
            break
        _, hi_pos, lo_pos, new_hi, new_lo = best
        packs[hi][hi_pos], packs[lo][lo_pos] = packs[lo][lo_pos], packs[hi][hi_pos]
        loads[hi] = new_hi
        loads[lo] = new_lo
    repaired = []
    for pack in packs:
        repaired.extend(pack)
    return repaired, ops


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
    _, group_ops = snake_pack_order(group_loads, num_nodes, num_groups // num_nodes)

    replica_counts, replica_ops = heap_replicate_overloaded(loads, num_replicas)
    physical_to_logical = []
    for expert_id, count in enumerate(replica_counts):
        physical_to_logical.extend([expert_id] * count)

    replica_loads = expert_replica_loads(loads, replica_counts)
    order, placement_ops = snake_pack_order(
        replica_loads, num_gpus, num_replicas // num_gpus
    )
    order, repair_ops = adaptive_repair(
        order, replica_loads, num_gpus, num_replicas // num_gpus
    )
    packed = [physical_to_logical[idx] for idx in order]
    return {
        "trace_id": trace["id"],
        "replica_counts": replica_counts,
        "physical_to_logical": packed,
        "operation_count": group_ops + replica_ops + placement_ops + repair_ops,
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
