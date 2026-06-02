#!/usr/bin/env python3
"""Evaluate EPLB placement artifacts without importing method code."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from eplb_common import gpu_loads_from_placement, imbalance_factor, load_traces, mean, write_json


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[rank]


def load_skew(trace: dict) -> float:
    loads = [float(x) for x in trace["loads"]]
    avg_load = mean(loads)
    return max(loads) / avg_load if avg_load > 0 else 0.0


def missing_operation_penalty(trace: dict) -> float:
    return float(trace["num_replicas"]) * 10.0


def artifact_outputs_by_trace(artifact: dict) -> tuple[dict[str, dict], set[str]]:
    outputs_by_trace: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for item in artifact.get("outputs", []):
        trace_id = item.get("trace_id")
        if not isinstance(trace_id, str):
            continue
        if trace_id in outputs_by_trace:
            duplicate_ids.add(trace_id)
            continue
        outputs_by_trace[trace_id] = item
    return outputs_by_trace, duplicate_ids


def safe_operation_count(item: dict | None, trace: dict) -> tuple[float, bool]:
    if item is None:
        return missing_operation_penalty(trace), False
    try:
        operation_count = float(item["operation_count"])
    except Exception:
        return missing_operation_penalty(trace), False
    if not math.isfinite(operation_count) or operation_count < 0:
        return missing_operation_penalty(trace), False
    return operation_count, True


def evaluate_artifact(traces: list[dict], artifact_path: Path) -> dict[str, float]:
    artifact = load_traces(artifact_path)
    outputs_by_trace, duplicate_ids = artifact_outputs_by_trace(artifact)

    imbalances: list[float] = []
    operation_counts: list[float] = []
    imbalances_by_family: dict[str, list[float]] = {}
    high_skew_imbalances: list[float] = []
    high_skew_operation_counts: list[float] = []
    skew_threshold = percentile([load_skew(trace) for trace in traces], 0.5)
    valid_count = 0
    complete_count = 0

    for trace in traces:
        trace_id = str(trace["id"])
        family = str(trace["family"])
        item = outputs_by_trace.get(trace_id)
        operation_count, has_valid_operation_count = safe_operation_count(item, trace)
        operation_counts.append(operation_count)
        is_high_skew = load_skew(trace) >= skew_threshold

        if item is None:
            imbalances.append(0.0)
            imbalances_by_family.setdefault(family, []).append(0.0)
            if is_high_skew:
                high_skew_imbalances.append(0.0)
                high_skew_operation_counts.append(operation_count)
            continue

        complete_count += 1
        if trace_id in duplicate_ids or not has_valid_operation_count:
            imbalances.append(0.0)
            imbalances_by_family.setdefault(family, []).append(0.0)
            if is_high_skew:
                high_skew_imbalances.append(0.0)
                high_skew_operation_counts.append(operation_count)
            continue

        try:
            gpu_loads = gpu_loads_from_placement(
                [float(x) for x in trace["loads"]],
                [int(x) for x in item["replica_counts"]],
                [int(x) for x in item["physical_to_logical"]],
                int(trace["num_gpus"]),
            )
        except Exception:
            imbalances.append(0.0)
            imbalances_by_family.setdefault(family, []).append(0.0)
            if is_high_skew:
                high_skew_imbalances.append(0.0)
                high_skew_operation_counts.append(operation_count)
            continue

        imbalance = imbalance_factor(gpu_loads)
        imbalances.append(imbalance)
        imbalances_by_family.setdefault(family, []).append(imbalance)
        if is_high_skew:
            high_skew_imbalances.append(imbalance)
            high_skew_operation_counts.append(operation_count)
        valid_count += 1

    denominator = max(1, len(traces))
    family_mean_imbalances = [mean(values) for values in imbalances_by_family.values()]
    return {
        "complete_fraction": complete_count / denominator,
        "family_imbalance_floor": min(family_mean_imbalances) if family_mean_imbalances else 0.0,
        "family_imbalance_spread": (
            max(family_mean_imbalances) - min(family_mean_imbalances)
            if family_mean_imbalances
            else 0.0
        ),
        "high_skew_mean_imbalance_factor": mean(high_skew_imbalances),
        "high_skew_mean_operation_count": mean(high_skew_operation_counts),
        "mean_imbalance_factor": mean(imbalances),
        "mean_operation_count": mean(operation_counts),
        "min_imbalance_factor": min(imbalances) if imbalances else 0.0,
        "num_traces": float(len(traces)),
        "p95_operation_count": percentile(operation_counts, 0.95),
        "valid_fraction": valid_count / denominator,
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
    outputs_dir = Path(args.outputs_dir)
    method_files = discover_artifacts(outputs_dir)
    if "greedy_deepseek" not in method_files:
        raise FileNotFoundError(
            "outputs/eplb/greedy_deepseek.json is required so work_efficiency_vs_greedy "
            "is computed against a fresh baseline. Run scripts/baseline.sh or scripts/reproduce.sh."
        )
    results = {}
    for name, path in method_files.items():
        results[name] = evaluate_artifact(traces, path)
    baseline_ops = results.get("greedy_deepseek", {}).get("mean_operation_count")
    if baseline_ops and baseline_ops > 0:
        for metrics in results.values():
            metrics["work_efficiency_vs_greedy"] = (
                baseline_ops / metrics["mean_operation_count"]
            ) * metrics["mean_imbalance_factor"]
    frontier_ops = results.get("frontier_reference_zigzag", {}).get("mean_operation_count")
    if frontier_ops and frontier_ops > 0:
        for metrics in results.values():
            metrics["work_efficiency_vs_frontier_reference"] = (
                frontier_ops / metrics["mean_operation_count"]
            ) * metrics["mean_imbalance_factor"]

    write_json(args.scores, {"experiments": {"eplb_moe_placement": {"results": results}}})


if __name__ == "__main__":
    main()
