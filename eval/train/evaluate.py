#!/usr/bin/env python3
"""Visible (train) evaluator for EPLB rebalancing policies.

Every ``outputs/eplb/<method>.py`` on disk is a submitted policy exposing the
paper's interface::

    rebalance_experts(weight[L, E], num_replicas, num_groups, num_nodes, num_gpus)
        -> (phy2log[L, R], log2phy[L, E, X], logcnt[L, E])

For each policy this evaluator starts a sandboxed harness process (harness.py)
and streams the visible train expert-load windows (data/eplb_train_windows.json)
to it one at a time: the policy
rebalances on window ``i`` and is scored on window ``i + 1``, which the policy
process never holds while it decides.  The evaluator then recomputes, from
the raw returned maps and the raw work counters, every reported number:

* validity of each placement (exact replica budget, every expert deployed,
  consistent maps, expert groups kept whole on equally loaded nodes),
* GPU load balance on the next window (mean GPU load / max GPU load, summed
  over layers as in the paper's evaluator),
* rebalance work measured by the harness, and the speedup against the frozen
  frontier-reference policy in this directory, run on the same windows.

A policy that fails anything (load error, crash, invalid map, guard violation,
non-determinism, work cap, or balance more than BALANCE_TOLERANCE below the
frontier reference) is scored as failed: worst value in every scored metric.

Usage: python eval/train/evaluate.py [--outputs-dir outputs/eplb] [--scores scoring/scores_train.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "eplb_moe_placement"
SLICE = "train"
WINDOWS_PATH = ROOT / "data" / "eplb_train_windows.json"
HARNESS_PATH = HERE / "harness.py"
ANCHOR_PATH = HERE / "anchor_frontier_reference.py"
ANCHOR_NAME = "frontier_reference_zigzag"

# Work units per measured event (one unit ~ one interpreted bytecode).
W_INSTRUCTION = 1.0
W_C_CALL = 1.0
W_PY_ELEMENT = 1.0
W_KERNEL = 400.0
W_TENSOR_ELEMENT = 0.1
COST_CAP = 5.0e8

BALANCE_TOLERANCE = 0.005
MAX_POLICY_BYTES = 64 * 1024
LOAD_TIMEOUT_S = 600.0
CALL_TIMEOUT_S = 1800.0

SCORED_WORST = {"speedup_vs_frontier_reference": 0.0, "balancedness": 0.0}


class HarnessError(RuntimeError):
    pass


class HarnessSession:
    """One sandboxed policy process, driven request by request."""

    def __init__(self, policy_path: Path, log_path: Path, workdir: Path):
        read_fd, write_fd = os.pipe()
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update(
            {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "HOME": str(workdir),
            }
        )
        self.log = log_path.open("wb")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-s",
                "-B",
                "-P",
                str(HARNESS_PATH),
                "--policy",
                str(policy_path),
                "--reply-fd",
                str(write_fd),
            ],
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=self.log,
            pass_fds=(write_fd,),
            env=env,
            cwd=str(workdir),
        )
        os.close(write_fd)
        self.fd = read_fd
        self.buffer = b""

    def request(self, payload: dict, timeout: float) -> dict:
        try:
            self.proc.stdin.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise HarnessError(f"policy process died ({exc})") from exc
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HarnessError(f"policy process exceeded {timeout:.0f}s safety timeout")
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.fd, 1 << 22)
            if not chunk:
                raise HarnessError("policy process exited unexpectedly")
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError("policy process sent a malformed reply") from exc
        if not isinstance(reply, dict):
            raise HarnessError("policy process sent a malformed reply")
        return reply

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(b'{"cmd":"exit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        os.close(self.fd)
        self.log.close()


def work_units(counts) -> float:
    if not isinstance(counts, list) or len(counts) != 5:
        raise HarnessError("missing work counters")
    values = []
    for value in counts:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessError("non-numeric work counter")
        if not math.isfinite(value) or value < 0:
            raise HarnessError("impossible work counter")
        values.append(float(value))
    cost = (
        W_INSTRUCTION * values[0]
        + W_C_CALL * values[1]
        + W_PY_ELEMENT * values[2]
        + W_KERNEL * values[3]
        + W_TENSOR_ELEMENT * values[4]
    )
    if cost <= 0:
        raise HarnessError("impossible (non-positive) rebalance work")
    return cost


def check_placement(reply: dict, cfg: dict, num_layers: int, num_experts: int) -> str | None:
    """Return None if the returned maps are a valid EPLB placement, else a reason."""
    num_replicas = cfg["num_replicas"]
    num_groups = cfg["num_groups"]
    num_nodes = cfg["num_nodes"]
    phy2log = reply.get("phy2log")
    logcnt = reply.get("logcnt")
    slots = reply.get("log2phy_slots")
    width = reply.get("log2phy_width")
    if not reply.get("log2phy_padding_ok", False):
        return "log2phy entries beyond each expert's replica count must be -1"
    if not (isinstance(phy2log, list) and isinstance(logcnt, list) and isinstance(slots, list)):
        return "missing maps"
    if not isinstance(width, int):
        return "missing log2phy width"
    if len(phy2log) != num_layers or len(logcnt) != num_layers or len(slots) != num_layers:
        return "maps have the wrong number of layers"
    hierarchical = num_groups % num_nodes == 0
    group_size = num_experts // num_groups
    slots_per_node = num_replicas // num_nodes
    groups_per_node = num_groups // num_nodes if hierarchical else 0
    for layer in range(num_layers):
        row = phy2log[layer]
        cnt_row = logcnt[layer]
        if not isinstance(row, list) or len(row) != num_replicas:
            return f"layer {layer}: phy2log must list exactly {num_replicas} physical experts"
        if not isinstance(cnt_row, list) or len(cnt_row) != num_experts:
            return f"layer {layer}: logcnt must list {num_experts} experts"
        positions: list[list[int]] = [[] for _ in range(num_experts)]
        for physical, expert in enumerate(row):
            if type(expert) is not int or not 0 <= expert < num_experts:
                return f"layer {layer}: invalid logical expert id {expert!r}"
            positions[expert].append(physical)
        layer_slots = slots[layer]
        if not isinstance(layer_slots, list) or len(layer_slots) != num_experts:
            return f"layer {layer}: log2phy has the wrong shape"
        for expert in range(num_experts):
            count = len(positions[expert])
            if count == 0:
                return f"layer {layer}: logical expert {expert} has no replica"
            if cnt_row[expert] != count:
                return f"layer {layer}: logcnt[{expert}]={cnt_row[expert]!r} but phy2log holds {count}"
            if count > width:
                return f"layer {layer}: log2phy is narrower than the replica count"
            listed = layer_slots[expert]
            if not isinstance(listed, list) or sorted(listed) != positions[expert]:
                return f"layer {layer}: log2phy disagrees with phy2log for expert {expert}"
        if hierarchical:
            node_of_group = [-1] * num_groups
            for physical, expert in enumerate(row):
                node = physical // slots_per_node
                group = expert // group_size
                if node_of_group[group] == -1:
                    node_of_group[group] = node
                elif node_of_group[group] != node:
                    return f"layer {layer}: expert group {group} is split across nodes"
            per_node = [0] * num_nodes
            for node in node_of_group:
                per_node[node] += 1
            if any(count != groups_per_node for count in per_node):
                return f"layer {layer}: each node must host exactly {groups_per_node} expert groups"
    return None


def balancedness(phy2log, logcnt, next_load, num_gpus: int) -> float:
    """Paper's imbalance factor: sum_l mean GPU load / sum_l max GPU load."""
    total_mean = 0.0
    total_max = 0.0
    for layer, row in enumerate(phy2log):
        per_gpu = len(row) // num_gpus
        gpu_load = [0.0] * num_gpus
        load_row = next_load[layer]
        cnt_row = logcnt[layer]
        for physical, expert in enumerate(row):
            gpu_load[physical // per_gpu] += load_row[expert] / cnt_row[expert]
        total_mean += sum(gpu_load) / num_gpus
        total_max += max(gpu_load)
    return total_mean / total_max if total_max > 0 else 1.0


def digest(reply: dict) -> str:
    payload = json.dumps(
        [reply.get("phy2log"), reply.get("logcnt"), reply.get("log2phy_slots")],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_policy(policy_path: Path, workload: dict, log_dir: Path) -> dict:
    """Run one policy over every rebalance window; return raw per-call records."""
    cfg = workload["config"]
    num_layers = workload["num_layers"]
    num_experts = workload["num_logical_experts"]
    sequences = workload["sequences"]
    total_calls = sum(len(seq["windows"]) - 1 for seq in sequences)
    record = {"calls": [], "cpu_ns": [], "failure": None, "total_calls": total_calls, "valid": 0}

    try:
        size = policy_path.stat().st_size
    except OSError as exc:
        record["failure"] = f"cannot read policy: {exc}"
        return record
    if size > MAX_POLICY_BYTES:
        record["failure"] = f"policy file is {size} bytes (limit {MAX_POLICY_BYTES})"
        return record

    workdir = Path(tempfile.mkdtemp(prefix="eplb_policy_"))
    policy_path = policy_path.resolve()
    session = HarnessSession(policy_path, log_dir.resolve() / f"{policy_path.stem}.log", workdir)
    try:
        digests: list[str] = []
        for seq in sequences:
            reply = session.request({"cmd": "load"}, LOAD_TIMEOUT_S)
            if not reply.get("ok"):
                raise HarnessError(str(reply.get("error", "load failed")))
            windows = seq["windows"]
            for index in range(len(windows) - 1):
                request = {"cmd": "call", "metered": True, "load": windows[index]["load"], **cfg}
                reply = session.request(request, CALL_TIMEOUT_S)
                if not reply.get("ok"):
                    raise HarnessError(
                        f"offset {seq['offset']} window {index}: {reply.get('error', 'call failed')}"
                    )
                cost = work_units(reply.get("counts"))
                if cost > COST_CAP:
                    raise HarnessError(f"offset {seq['offset']} window {index}: work cap exceeded")
                reason = check_placement(reply, cfg, num_layers, num_experts)
                if reason is not None:
                    raise HarnessError(f"offset {seq['offset']} window {index}: invalid placement: {reason}")
                record["valid"] += 1
                bal = balancedness(
                    reply["phy2log"], reply["logcnt"], windows[index + 1]["load"], cfg["num_gpus"]
                )
                record["calls"].append(
                    {"offset": seq["offset"], "window": index, "cost": cost, "balancedness": bal,
                     "counts": reply["counts"]}
                )
                digests.append(digest(reply))
        # Unmetered replay: outputs must be identical (determinism) and the
        # host CPU time is kept as an unscored diagnostic.
        position = 0
        for seq in sequences:
            reply = session.request({"cmd": "load"}, LOAD_TIMEOUT_S)
            if not reply.get("ok"):
                raise HarnessError(str(reply.get("error", "reload failed")))
            windows = seq["windows"]
            for index in range(len(windows) - 1):
                request = {"cmd": "call", "metered": False, "load": windows[index]["load"], **cfg}
                reply = session.request(request, CALL_TIMEOUT_S)
                if not reply.get("ok"):
                    raise HarnessError(f"unmetered replay failed: {reply.get('error', 'call failed')}")
                if digest(reply) != digests[position]:
                    raise HarnessError("policy output is not deterministic (metered and unmetered runs differ)")
                cpu_ns = reply.get("cpu_ns")
                record["cpu_ns"].append(float(cpu_ns) if isinstance(cpu_ns, (int, float)) else 0.0)
                position += 1
    except HarnessError as exc:
        record["failure"] = str(exc)
    finally:
        session.close()
    return record


def summarize(record: dict, anchor: dict | None) -> tuple[dict, str | None]:
    calls = record["calls"]
    total = record["total_calls"]
    failure = record["failure"]
    costs = [c["cost"] for c in calls]
    bals = [c["balancedness"] for c in calls]
    mean_cost = sum(costs) / len(costs) if costs else COST_CAP
    mean_bal = sum(bals) / len(bals) if bals else 0.0
    cpu = record["cpu_ns"]
    metrics = {
        "mean_rebalance_cost": mean_cost,
        "worst_window_balancedness": min(bals) if bals else 0.0,
        "valid_fraction": record["valid"] / total if total else 0.0,
        "diagnostic_cpu_ms_per_rebalance": (sum(cpu) / len(cpu) / 1e6) if cpu else 0.0,
        "num_rebalances": float(total),
    }
    if anchor is not None:
        metrics["balance_change_vs_frontier_reference"] = mean_bal - anchor["balancedness"]
        if failure is None and mean_bal < anchor["balancedness"] - BALANCE_TOLERANCE:
            failure = (
                f"balance constraint: balancedness {mean_bal:.4f} is more than "
                f"{BALANCE_TOLERANCE} below the frontier reference ({anchor['balancedness']:.4f})"
            )
    if failure is None and len(calls) != total:
        failure = "not every rebalance window was scored"
    if failure is None:
        metrics["balancedness"] = mean_bal
        metrics["speedup_vs_frontier_reference"] = (
            anchor["mean_cost"] / mean_cost if anchor is not None else 1.0
        )
        metrics["run_failed"] = 0.0
    else:
        metrics.update(SCORED_WORST)
        metrics["worst_window_balancedness"] = 0.0
        metrics["run_failed"] = 1.0
    return metrics, failure


def discover_policies(outputs_dir: Path) -> dict[str, Path]:
    if not outputs_dir.is_dir():
        return {}
    found = {}
    for path in sorted(outputs_dir.glob("*.py")):
        if path.name.startswith(("_", ".")) or not path.is_file():
            continue
        found[path.stem] = path
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs" / "eplb"))
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--details", default=None, help="optional per-call record dump (JSON)")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    workload = json.loads(WINDOWS_PATH.read_text())
    log_dir = Path(args.log_dir) if args.log_dir else Path(tempfile.mkdtemp(prefix="eplb_logs_"))
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{SLICE}] running frozen anchor {ANCHOR_NAME}", flush=True)
    anchor_record = run_policy(ANCHOR_PATH, workload, log_dir)
    if anchor_record["failure"] is not None:
        raise SystemExit(f"anchor policy failed: {anchor_record['failure']} (logs in {log_dir})")
    anchor = {
        "mean_cost": sum(c["cost"] for c in anchor_record["calls"]) / len(anchor_record["calls"]),
        "balancedness": sum(c["balancedness"] for c in anchor_record["calls"]) / len(anchor_record["calls"]),
    }

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    details: dict[str, dict] = {"_anchor": anchor_record}
    for name, path in discover_policies(Path(args.outputs_dir)).items():
        started = time.monotonic()
        record = run_policy(path, workload, log_dir)
        metrics, failure = summarize(record, anchor)
        results[name] = metrics
        details[name] = record
        if failure is not None:
            failures[name] = failure
        status = "FAILED: " + failure if failure else "ok"
        print(
            f"[{SLICE}] {name}: speedup={metrics['speedup_vs_frontier_reference']:.4f} "
            f"balancedness={metrics['balancedness']:.4f} ({time.monotonic() - started:.0f}s) {status}",
            flush=True,
        )

    payload = {
        "_slice": {
            "name": SLICE,
            "windows": "visible (even) MoE layers of the paper's trace, 4 window offsets",
            "num_rebalances_per_method": sum(len(s["windows"]) - 1 for s in workload["sequences"]),
        },
        "_anchor": {"name": ANCHOR_NAME, **anchor},
        "_work_measure": {
            "units_per_bytecode": W_INSTRUCTION,
            "units_per_c_call": W_C_CALL,
            "units_per_scanned_python_element": W_PY_ELEMENT,
            "units_per_torch_kernel": W_KERNEL,
            "units_per_tensor_element": W_TENSOR_ELEMENT,
            "per_rebalance_cap": COST_CAP,
            "balance_tolerance_vs_anchor": BALANCE_TOLERANCE,
        },
        "_failures": failures,
        "experiments": {EXPERIMENT: {"results": results}},
    }
    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    if args.details:
        Path(args.details).write_text(json.dumps(details, separators=(",", ":")) + "\n")
    print(f"[{SLICE}] wrote {out} ({len(results)} methods, {len(failures)} failed)", flush=True)


if __name__ == "__main__":
    main()
