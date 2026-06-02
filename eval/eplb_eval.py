#!/usr/bin/env python3
"""Evaluate EPLB placement artifacts without importing method code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from eplb_common import gpu_loads_from_placement, imbalance_factor, load_traces, mean, write_json


def evaluate_artifact(traces_by_id: dict[str, dict], artifact_path: Path) -> dict[str, float]:
    artifact = load_traces(artifact_path)
    imbalances = []
    ops = []
    violations = 0
    for item in artifact["outputs"]:
        trace = traces_by_id[item["trace_id"]]
        try:
            gpu_loads = gpu_loads_from_placement(
                [float(x) for x in trace["loads"]],
                [int(x) for x in item["replica_counts"]],
                [int(x) for x in item["physical_to_logical"]],
                int(trace["num_gpus"]),
            )
            imbalances.append(imbalance_factor(gpu_loads))
        except Exception:
            violations += 1
            imbalances.append(0.0)
        ops.append(float(item["operation_count"]))

    return {
        "mean_imbalance_factor": mean(imbalances),
        "mean_operation_count": mean(ops),
        "valid_fraction": 1.0 - violations / max(1, len(artifact["outputs"])),
        "num_traces": float(len(artifact["outputs"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/eplb_traces.json")
    parser.add_argument("--outputs-dir", default="outputs/eplb")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    traces = load_traces(args.traces)["traces"]
    traces_by_id = {trace["id"]: trace for trace in traces}
    outputs_dir = Path(args.outputs_dir)
    method_files = {
        "greedy_deepseek": outputs_dir / "greedy_deepseek.json",
        "evolved_zigzag": outputs_dir / "evolved_zigzag.json",
    }
    results = {}
    for name, path in method_files.items():
        if path.exists():
            results[name] = evaluate_artifact(traces_by_id, path)
    baseline_ops = results.get("greedy_deepseek", {}).get("mean_operation_count")
    if baseline_ops and baseline_ops > 0:
        for metrics in results.values():
            metrics["work_efficiency_vs_greedy"] = baseline_ops / metrics["mean_operation_count"]

    write_json(args.scores, {"experiments": {"eplb_moe_placement": {"results": results}}})


if __name__ == "__main__":
    main()
