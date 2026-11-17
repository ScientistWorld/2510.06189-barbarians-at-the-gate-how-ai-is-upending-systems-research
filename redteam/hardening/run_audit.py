#!/usr/bin/env python3
"""Hardening audit: run every case through the hidden-test evaluator.

1. combined hidden-test run: reference + baselines + every case in cases/,
   each as one policy file in a fresh outputs directory (same code path as
   ``bash scripts/evaluate_test.sh``, which scores every policy on disk);
2. visible (train) run for the honest variants (train/test parity);
3. isolated cases (cases/isolated/), each in its own hidden-test run, for
   attacks that would take the evaluator down if a guard were missing;
4. disk-level shortcuts that do not live in a policy file (edit/remove the
   comparator policy, flatten the visible data, forge score files, drop the
   train roster), each followed by a hidden-test run;
5. two more independent hidden-test runs of the reference and the frontier
   baseline (run-to-run noise of the instruction counts);
6. cohort ranks with the rule in scoring/reference.json, paired per-pair
   statistics against the reference, and a calibration of the instruction
   count against CPU time across implementation styles.

Results go to redteam/hardening/results/.  Usage: python redteam/hardening/run_audit.py
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = HERE / "cases"
ISOLATED = CASES / "isolated"
RESULTS = HERE / "results"
WORK = HERE / ".work"
EXP = "eplb_moe_placement"
REFERENCE = "evolved_zigzag"
HONEST = [
    "evolved_zigzag", "frontier_reference_zigzag", "greedy_deepseek",
    "better_compact_log2phy", "better_lean_replication", "weaker_without_snake",
    "brute_force_sampled_placements", "simple_static_placement",
    "probe_python_lists", "probe_python_slicing", "probe_numpy_backend",
]
PUBLISHED = {
    "evolved_zigzag": ROOT / "method" / "eplb_evolved.py",
    "frontier_reference_zigzag": ROOT / "baseline" / "eplb_frontier_reference.py",
    "greedy_deepseek": ROOT / "baseline" / "eplb_greedy_deepseek.py",
}
SCORED = ("speedup_vs_frontier_reference", "balancedness")
ISOLATED_TIMEOUT_S = 3600


def eval_hash() -> str:
    digest = hashlib.sha256()
    for rel in ("eval/test/evaluate.py", "eval/test/harness.py", "eval/test/anchor_frontier_reference.py",
                "eval/test/test_windows.json", "eval/train/evaluate.py", "eval/train/harness.py",
                "data/eplb_train_windows.json", "method/eplb_evolved.py",
                "baseline/eplb_frontier_reference.py", "baseline/eplb_greedy_deepseek.py"):
        digest.update((ROOT / rel).read_bytes())
    for case in sorted(CASES.rglob("*.py")):
        digest.update(case.read_bytes())
    return digest.hexdigest()


STAGES = RESULTS / "stages.json"
EVAL_HASH = ""


def stage_done(name: str) -> bool:
    if not STAGES.exists():
        return False
    return json.loads(STAGES.read_text()).get(name) == EVAL_HASH


def mark_done(name: str) -> None:
    done = json.loads(STAGES.read_text()) if STAGES.exists() else {}
    done[name] = EVAL_HASH
    STAGES.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")


def log(msg: str) -> None:
    print(f"[audit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def run_eval(slice_: str, outputs: Path, scores: Path, details: Path | None, tag: str,
             timeout: float | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "eval" / slice_ / "evaluate.py"),
           "--outputs-dir", str(outputs), "--scores", str(scores),
           "--log-dir", str(WORK / "logs" / tag)]
    if details is not None:
        cmd += ["--details", str(details)]
    log(f"{tag}: {' '.join(cmd[1:3])} on {outputs.name}")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = -999
    log(f"{tag}: exit {code} after {time.monotonic() - started:.0f}s")
    return code


def results_of(scores: Path) -> dict:
    payload = json.loads(scores.read_text())
    return payload["experiments"][EXP]["results"]


def case_path(name: str) -> Path:
    if name in PUBLISHED:
        return PUBLISHED[name]
    path = CASES / f"{name}.py"
    return path if path.exists() else ISOLATED / f"{name}.py"


def stage_outputs(dest: Path, names: list[str]) -> None:
    for name in names:
        shutil.copy(case_path(name), dest / f"{name}.py")


def scored_metrics() -> dict[str, dict]:
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())
    exp = ref["experiments"][EXP]
    weight = float(exp.get("weight", 1.0))
    out = {}
    for name, spec in exp["metrics"].items():
        coef = abs(float(spec.get("coefficient", 0.0)))
        if coef > 0:
            out[name] = {"coef": coef * weight, "higher": bool(spec["higher_is_better"])}
    return out


def cohort_ranks(results: dict, cohort: list[str], metrics: dict) -> dict:
    present = [m for m in cohort if m in results]
    scales = {}
    for metric in metrics:
        values = [results[m][metric] for m in present if isinstance(results[m].get(metric), (int, float))]
        scales[metric] = statistics.pstdev(values) if len(values) > 1 else 0.0
    ref = results[REFERENCE]
    ranks = {}
    for m in present:
        total = 0.0
        terms = {}
        for metric, spec in metrics.items():
            a, b = results[m].get(metric), ref.get(metric)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or scales[metric] == 0:
                continue
            delta = (a - b) if spec["higher"] else (b - a)
            term = spec["coef"] * delta / scales[metric]
            terms[metric] = term
            total += term
        ranks[m] = {"rank_delta": total, "terms": terms}
    return {"scales": scales, "ranks": ranks}


def paired(details: dict, name: str) -> dict | None:
    ref = details.get(REFERENCE)
    var = details.get(name)
    if not ref or not var or var.get("failure") or len(var["calls"]) != len(ref["calls"]):
        return None
    ratios, diffs = [], []
    for a, b in zip(ref["calls"], var["calls"]):
        ratios.append(a["instructions"] / b["instructions"])
        diffs.append(b["balancedness"] - a["balancedness"])
    n = len(diffs)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    return {
        "n_pairs": n,
        "work_ratio_ref_over_variant_mean": statistics.fmean(ratios),
        "work_ratio_min": min(ratios),
        "work_ratio_max": max(ratios),
        "balance_diff_mean": statistics.fmean(diffs),
        "balance_diff_se": sd / math.sqrt(n) if n else 0.0,
        "balance_diff_min": min(diffs),
        "balance_diff_max": max(diffs),
        "pairs_variant_cheaper": sum(r > 1 for r in ratios),
        "pairs_variant_better_balanced": sum(d > 0 for d in diffs),
        "pairs_variant_worse_balanced": sum(d < 0 for d in diffs),
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        for i, j in enumerate(order):
            r[j] = float(i)
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Disk-level shortcuts and isolated cases
# ---------------------------------------------------------------------------
def scored_of(res: dict, name: str) -> dict:
    mine = res.get(name, {})
    return {k: mine.get(k) for k in SCORED + ("run_failed", "mean_rebalance_instructions")}


def disk_level_runs(combined: dict) -> dict:
    disk_file = RESULTS / "disk_level_shortcuts.json"
    out = json.loads(disk_file.read_text()) if disk_file.exists() and stage_done("disk_partial") else {}
    ref_combined = combined[REFERENCE]

    def save() -> None:
        disk_file.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        mark_done("disk_partial")

    def compare(tag: str, res: dict, extra: dict | None = None) -> None:
        mine = res.get(REFERENCE, {})
        out[tag] = {
            "reference_in_this_run": scored_of(res, REFERENCE),
            "reference_relative_change": {
                k: (mine.get(k) - ref_combined[k]) / ref_combined[k]
                if isinstance(mine.get(k), (int, float)) and ref_combined.get(k) else None
                for k in SCORED
            },
            **(extra or {}),
        }
        log(f"{tag}: reference {out[tag]['reference_in_this_run']}")
        save()

    for tag, fn in (("shortcut_inflate_baseline", _inflate), ("shortcut_remove_frontier", _remove),
                    ("shortcut_flatten_workload", _flatten), ("shortcut_forged_summary", _forged),
                    ("shortcut_omit_method_from_roster", _roster)):
        if tag not in out:
            fn(compare)
    return out


def _inflate(compare) -> None:
    d = fresh_dir(WORK / "inflate_frontier")
    stage_outputs(d, [REFERENCE])
    slow = PUBLISHED["frontier_reference_zigzag"].read_text().replace(
        "    num_layers, num_logical_experts = weight.shape\n    weight = weight.float().cpu()\n",
        "    num_layers, num_logical_experts = weight.shape\n    weight = weight.float().cpu()\n"
        "    for _ in range(200000):\n        pass\n", 1)
    (d / "frontier_reference_zigzag.py").write_text(slow)
    run_eval("test", d, d / "scores.json", None, "inflate_frontier")
    res = results_of(d / "scores.json")
    compare("shortcut_inflate_baseline", res,
            {"edited_frontier": scored_of(res, "frontier_reference_zigzag")})


def _remove(compare) -> None:
    d = fresh_dir(WORK / "remove_frontier")
    stage_outputs(d, [REFERENCE])
    run_eval("test", d, d / "scores.json", None, "remove_frontier")
    compare("shortcut_remove_frontier", results_of(d / "scores.json"))


def _flatten(compare) -> None:
    train_file = ROOT / "data" / "eplb_train_windows.json"
    backup = WORK / "train_windows.backup.json"
    shutil.copy(train_file, backup)
    try:
        payload = json.loads(train_file.read_text())
        for pair in payload["pairs"]:
            for key in ("load", "next_load"):
                pair[key] = [[1] * len(row) for row in pair[key]]
        train_file.write_text(json.dumps(payload))
        d = fresh_dir(WORK / "flatten")
        stage_outputs(d, [REFERENCE])
        run_eval("test", d, d / "scores.json", None, "flatten")
        compare("shortcut_flatten_workload", results_of(d / "scores.json"))
    finally:
        shutil.copy(backup, train_file)


def _forged(compare) -> None:
    d = fresh_dir(WORK / "forged")
    stage_outputs(d, [REFERENCE])
    forged = {"experiments": {EXP: {"results": {REFERENCE: {
        "speedup_vs_frontier_reference": 1e9, "balancedness": 1.0, "run_failed": 0.0}}}}}
    (d / "scores.json").write_text(json.dumps(forged))
    run_eval("test", d, d / "scores.json", None, "forged")
    compare("shortcut_forged_summary", results_of(d / "scores.json"))


def _roster(compare) -> None:
    train_scores = ROOT / "scoring" / "scores_train.json"
    moved = WORK / "scores_train.moved.json"
    had = train_scores.exists()
    if had:
        shutil.move(train_scores, moved)
    try:
        d = fresh_dir(WORK / "roster")
        stage_outputs(d, [REFERENCE])
        run_eval("test", d, d / "scores.json", None, "roster")
        compare("shortcut_omit_method_from_roster", results_of(d / "scores.json"))
    finally:
        if had:
            shutil.move(moved, train_scores)


def isolated_runs(combined: dict) -> dict:
    out_file = RESULTS / "isolated_cases.json"
    out = json.loads(out_file.read_text()) if out_file.exists() and stage_done("isolated_partial") else {}
    for case in sorted(ISOLATED.glob("*.py")):
        name = case.stem
        if name in out:
            continue
        d = fresh_dir(WORK / f"isolated_{name}")
        stage_outputs(d, [REFERENCE, name])
        code = run_eval("test", d, d / "scores.json", None, f"isolated_{name}", timeout=ISOLATED_TIMEOUT_S)
        res = results_of(d / "scores.json") if (d / "scores.json").exists() else {}
        failures = json.loads((d / "scores.json").read_text()).get("_failures", {}) if res else {}
        out[name] = {
            "evaluator_exit_code": code,
            "case": scored_of(res, name),
            "case_failure": failures.get(name),
            "reference_in_this_run": scored_of(res, REFERENCE),
            "reference_unchanged": all(res.get(REFERENCE, {}).get(k) == combined[REFERENCE][k] for k in SCORED),
        }
        log(f"isolated {name}: exit {code}, case {out[name]['case']}")
        out_file.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        mark_done("isolated_partial")
    return out


def repeat_runs(combined: dict, details: dict) -> dict:
    out_file = RESULTS / "repeat_runs.json"
    if out_file.exists() and stage_done("repeats"):
        return json.loads(out_file.read_text())
    runs = []
    for k in (1, 2):
        d = fresh_dir(WORK / f"repeat_{k}")
        stage_outputs(d, [REFERENCE, "frontier_reference_zigzag"])
        run_eval("test", d, d / "scores.json", d / "details.json", f"repeat_{k}")
        res = results_of(d / "scores.json")
        det = json.loads((d / "details.json").read_text())
        runs.append({"scores": {m: scored_of(res, m) for m in (REFERENCE, "frontier_reference_zigzag")},
                     "instructions": {m: [c["instructions"] for c in det[m]["calls"]]
                                      for m in (REFERENCE, "frontier_reference_zigzag", "_anchor")}})
    base = {m: [c["instructions"] for c in details[m]["calls"]] for m in (REFERENCE, "frontier_reference_zigzag", "_anchor")}
    spread = {}
    for m in base:
        rel = []
        for run in runs:
            rel.extend(abs(a - b) / b for a, b in zip(run["instructions"][m], base[m]))
        spread[m] = {"max_relative_per_pair_difference": max(rel) if rel else None}
    speedups = [combined[REFERENCE]["speedup_vs_frontier_reference"]] + [
        r["scores"][REFERENCE]["speedup_vs_frontier_reference"] for r in runs]
    out = {"runs": runs, "per_pair_instruction_spread_vs_combined": spread,
           "reference_speedup_in_three_runs": speedups,
           "reference_speedup_max_minus_min": max(speedups) - min(speedups),
           "reference_balancedness_in_three_runs": [combined[REFERENCE]["balancedness"]] + [
               r["scores"][REFERENCE]["balancedness"] for r in runs]}
    out_file.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    mark_done("repeats")
    return out


def main() -> None:
    global EVAL_HASH
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    EVAL_HASH = eval_hash()
    case_names = sorted(p.stem for p in CASES.glob("*.py"))
    all_names = list(PUBLISHED) + case_names

    details_test = RESULTS / "combined_details_test.json"
    if not stage_done("combined_test"):
        combined_dir = fresh_dir(WORK / "combined")
        stage_outputs(combined_dir, all_names)
        run_eval("test", combined_dir, RESULTS / "combined_scores_test.json", details_test, "combined_test")
        mark_done("combined_test")
    combined_payload = json.loads((RESULTS / "combined_scores_test.json").read_text())
    combined = combined_payload["experiments"][EXP]["results"]
    details = json.loads(details_test.read_text())

    details_train = RESULTS / "honest_details_train.json"
    if not stage_done("honest_train"):
        train_dir = fresh_dir(WORK / "train")
        stage_outputs(train_dir, HONEST)
        run_eval("train", train_dir, RESULTS / "honest_scores_train.json", details_train, "honest_train")
        mark_done("honest_train")
    train = results_of(RESULTS / "honest_scores_train.json")

    isolated = isolated_runs(combined)
    disk = disk_level_runs(combined)
    repeats = repeat_runs(combined, details)

    metrics = scored_metrics()
    honest_present = [m for m in HONEST if m in combined]
    ok = [m for m in combined if not combined[m]["run_failed"]]
    calibration = {
        m: {"instructions": combined[m]["mean_rebalance_instructions"],
            "cpu_ms": combined[m]["diagnostic_cpu_ms_per_rebalance"],
            "instructions_per_cpu_ms": combined[m]["mean_rebalance_instructions"]
            / max(combined[m]["diagnostic_cpu_ms_per_rebalance"], 1e-9)}
        for m in ok
    }
    ipms = [v["instructions_per_cpu_ms"] for v in calibration.values()]
    summary = {
        "evaluator_hash": EVAL_HASH,
        "scored_metrics": metrics,
        "anchor": combined_payload.get("_anchor"),
        "measurement": combined_payload.get("_measurement"),
        "failures": combined_payload.get("_failures"),
        "test": combined,
        "train_honest": train,
        "ranks_honest_cohort": cohort_ranks(combined, honest_present, metrics),
        "ranks_full_cohort": cohort_ranks(combined, list(combined), metrics),
        "paired_vs_reference": {m: paired(details, m) for m in combined if m != REFERENCE},
        "isolated_cases": isolated,
        "disk_level_shortcuts": disk,
        "repeat_runs": repeats,
        "work_measure_calibration": {
            "methods": calibration,
            "spearman_instructions_vs_cpu": spearman([v["instructions"] for v in calibration.values()],
                                                     [v["cpu_ms"] for v in calibration.values()])
            if len(ok) > 2 else None,
            "instructions_per_cpu_ms_min": min(ipms) if ipms else None,
            "instructions_per_cpu_ms_max": max(ipms) if ipms else None,
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = ["| method | speedup | balancedness | instructions | cpu ms | failed | rank Δ honest | rank Δ full | reason |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    rh = summary["ranks_honest_cohort"]["ranks"]
    rf = summary["ranks_full_cohort"]["ranks"]
    for m in all_names:
        if m not in combined:
            continue
        r = combined[m]
        reason = (summary["failures"] or {}).get(m, "")
        lines.append(
            f"| {m} | {r['speedup_vs_frontier_reference']:.4f} | {r['balancedness']:.4f} | "
            f"{r['mean_rebalance_instructions']:.4g} | {r['diagnostic_cpu_ms_per_rebalance']:.2f} | "
            f"{int(r['run_failed'])} | {rh.get(m, {}).get('rank_delta', float('nan')):.3f} | "
            f"{rf[m]['rank_delta']:.3f} | {reason[:110]} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    log("wrote results/summary.json and results/summary.md")
    print("\n".join(lines))
    print(json.dumps({"isolated": isolated, "repeats": {k: v for k, v in repeats.items() if k != "runs"},
                      "calibration": {k: v for k, v in summary["work_measure_calibration"].items() if k != "methods"}},
                     indent=1))


if __name__ == "__main__":
    main()
