#!/usr/bin/env python3
"""Visible EPLB train-slice evaluator.

This scorer is method-agnostic: it reads placement artifacts from outputs/eplb
and evaluates only the train slice of the deterministic trace pool.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))

from eplb_common import gpu_loads_from_placement, imbalance_factor, load_traces, mean, write_json

EXPERIMENT = "eplb_moe_placement"
SLICE = "train"


def split_traces(traces: list[dict]) -> list[dict]:
    """Return every even-indexed trace within each family, sorted by trace id."""
    by_family: dict[str, list[dict]] = {}
    for trace in traces:
        by_family.setdefault(str(trace["family"]), []).append(trace)

    selected: list[dict] = []
    for family in sorted(by_family):
        family_traces = sorted(by_family[family], key=lambda item: str(item["id"]))
        selected.extend(trace for index, trace in enumerate(family_traces) if index % 2 == 0)
    return sorted(selected, key=lambda item: str(item["id"]))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[rank]


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


def evaluate_artifact(selected_traces: list[dict], artifact_path: Path) -> dict[str, float]:
    artifact = load_traces(artifact_path)
    outputs_by_trace, duplicate_ids = artifact_outputs_by_trace(artifact)

    imbalances: list[float] = []
    operation_counts: list[float] = []
    valid_count = 0
    complete_count = 0

    for trace in selected_traces:
        trace_id = str(trace["id"])
        item = outputs_by_trace.get(trace_id)
        operation_count, has_valid_operation_count = safe_operation_count(item, trace)
        operation_counts.append(operation_count)

        if item is None:
            imbalances.append(0.0)
            continue

        complete_count += 1
        if trace_id in duplicate_ids or not has_valid_operation_count:
            imbalances.append(0.0)
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
            continue

        imbalances.append(imbalance_factor(gpu_loads))
        valid_count += 1

    denominator = max(1, len(selected_traces))
    return {
        "complete_fraction": complete_count / denominator,
        "mean_imbalance_factor": mean(imbalances),
        "mean_operation_count": mean(operation_counts),
        "min_imbalance_factor": min(imbalances) if imbalances else 0.0,
        "num_traces": float(len(selected_traces)),
        "p95_operation_count": percentile(operation_counts, 0.95),
        "valid_fraction": valid_count / denominator,
    }


def discover_artifacts(outputs_dir: Path, method_names: Iterable[str] | None = None) -> dict[str, Path]:
    requested = list(method_names) if method_names is not None else None
    artifacts: dict[str, Path] = {}
    for path in sorted(outputs_dir.glob("*.json")):
        artifact = load_traces(path)
        method_name = artifact.get("method", path.stem)
        if not isinstance(method_name, str) or not method_name:
            raise ValueError(f"{path} has no usable method name")
        if method_name in artifacts:
            raise ValueError(f"multiple artifacts declare method {method_name!r}")
        artifacts[method_name] = path

    if requested is not None:
        missing = [name for name in requested if name not in artifacts]
        if missing:
            raise FileNotFoundError(
                "missing placement artifacts for train-scored methods: " + ", ".join(missing)
            )
        artifacts = {name: artifacts[name] for name in requested}

    return artifacts


def add_relative_efficiency_metrics(results: dict[str, dict[str, float]]) -> None:
    baseline_ops = results.get("greedy_deepseek", {}).get("mean_operation_count")
    if baseline_ops and baseline_ops > 0:
        for metrics in results.values():
            denominator = metrics["mean_operation_count"]
            metrics["work_efficiency_vs_greedy"] = baseline_ops / denominator if denominator > 0 else 0.0

    frontier_ops = results.get("frontier_reference_zigzag", {}).get("mean_operation_count")
    if frontier_ops and frontier_ops > 0:
        for metrics in results.values():
            denominator = metrics["mean_operation_count"]
            metrics["work_efficiency_vs_frontier_reference"] = (
                frontier_ops / denominator if denominator > 0 else 0.0
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/eplb_traces.json")
    parser.add_argument("--outputs-dir", default="outputs/eplb")
    parser.add_argument("--scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    all_traces = load_traces(args.traces)["traces"]
    selected_traces = split_traces(all_traces)
    method_files = discover_artifacts(Path(args.outputs_dir))
    if "greedy_deepseek" not in method_files:
        raise FileNotFoundError(
            "outputs/eplb/greedy_deepseek.json is required so work_efficiency_vs_greedy "
            "is computed against a fresh baseline. Run scripts/baseline.sh or scripts/reproduce.sh."
        )

    results = {
        name: evaluate_artifact(selected_traces, path)
        for name, path in method_files.items()
    }
    add_relative_efficiency_metrics(results)

    write_json(
        args.scores,
        {
            "_slice": {
                "name": SLICE,
                "split": "even trace index within each family",
                "num_traces": len(selected_traces),
            },
            "experiments": {EXPERIMENT: {"results": results}},
        },
    )


if __name__ == "__main__":
    main()
