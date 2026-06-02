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


def discover_artifacts(outputs_dir: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for path in sorted(outputs_dir.glob("*.json")):
        artifact = load_traces(path)
        method_name = artifact.get("method", path.stem)
        if not isinstance(method_name, str) or not method_name:
            raise ValueError(f"{path} has no usable method name")
        artifacts[method_name] = path
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/eplb_traces.json")
    parser.add_argument("--outputs-dir", default="outputs/eplb")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    traces = load_traces(args.traces)["traces"]
    traces_by_id = {trace["id"]: trace for trace in traces}
    outputs_dir = Path(args.outputs_dir)
    method_files = discover_artifacts(outputs_dir)
    if "greedy_deepseek" not in method_files:
        raise FileNotFoundError(
            "outputs/eplb/greedy_deepseek.json is required so work_efficiency_vs_greedy "
            "is computed against a fresh baseline. Run scripts/baseline.sh or scripts/reproduce.sh."
        )
    results = {}
    for name, path in method_files.items():
        results[name] = evaluate_artifact(traces_by_id, path)
    baseline_ops = results.get("greedy_deepseek", {}).get("mean_operation_count")
    if baseline_ops and baseline_ops > 0:
        for metrics in results.values():
            metrics["work_efficiency_vs_greedy"] = baseline_ops / metrics["mean_operation_count"]
    frontier_ops = results.get("frontier_reference_zigzag", {}).get("mean_operation_count")
    if frontier_ops and frontier_ops > 0:
        for metrics in results.values():
            metrics["work_efficiency_vs_frontier_reference"] = (
                frontier_ops / metrics["mean_operation_count"]
            )

    write_json(args.scores, {"experiments": {"eplb_moe_placement": {"results": results}}})


if __name__ == "__main__":
    main()
