#!/usr/bin/env python3
"""Hardening audit: run every case through the hidden-test evaluator.

1. combined hidden-test run: reference + baselines + every case in cases/,
   each as one policy file in a fresh outputs directory (same code path as
   ``bash scripts/evaluate_test.sh``, which scores every policy on disk);
2. visible (train) run for the honest variants (train/test parity);
3. disk-level shortcuts that do not live in a policy file (edit/remove the
   comparator policy, flatten the visible data, forge score files, drop the
   train roster), each followed by a hidden-test run;
4. cohort ranks with the rule in scoring/reference.json, paired per-window
   statistics against the reference, and a work-measure calibration check.

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
RESULTS = HERE / "results"
WORK = HERE / ".work"
EXP = "eplb_moe_placement"
REFERENCE = "evolved_zigzag"
HONEST = [
    "evolved_zigzag", "frontier_reference_zigzag", "greedy_deepseek",
    "better_compact_log2phy", "better_lean_replication", "weaker_without_snake",
    "brute_force_sampled_placements", "simple_static_placement", "probe_python_lists",
]
PUBLISHED = {
    "evolved_zigzag": ROOT / "method" / "eplb_evolved.py",
    "frontier_reference_zigzag": ROOT / "baseline" / "eplb_frontier_reference.py",
    "greedy_deepseek": ROOT / "baseline" / "eplb_greedy_deepseek.py",
}


def eval_hash() -> str:
    digest = hashlib.sha256()
    for rel in ("eval/test/evaluate.py", "eval/test/harness.py", "eval/test/anchor_frontier_reference.py",
                "eval/test/test_windows.json", "eval/train/evaluate.py", "eval/train/harness.py",
                "data/eplb_train_windows.json"):
        digest.update((ROOT / rel).read_bytes())
    for case in sorted(CASES.glob("*.py")):
        digest.update(case.read_bytes())
    return digest.hexdigest()


STAGES = RESULTS / "stages.json"


def stage_done(name: str) -> bool:
    if not STAGES.exists():
        return False
    return json.loads(STAGES.read_text()).get(name) == EVAL_HASH


def mark_done(name: str) -> None:
    done = json.loads(STAGES.read_text()) if STAGES.exists() else {}
    done[name] = EVAL_HASH
    STAGES.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")


EVAL_HASH = ""


def log(msg: str) -> None:
    print(f"[audit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def run_eval(slice_: str, outputs: Path, scores: Path, details: Path | None, tag: str) -> int:
    cmd = [sys.executable, str(ROOT / "eval" / slice_ / "evaluate.py"),
           "--outputs-dir", str(outputs), "--scores", str(scores),
           "--log-dir", str(WORK / "logs" / tag)]
    if details is not None:
        cmd += ["--details", str(details)]
    log(f"{tag}: {' '.join(cmd[1:3])} on {outputs.name}")
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT)
    log(f"{tag}: exit {proc.returncode} after {time.monotonic() - started:.0f}s")
    return proc.returncode


def results_of(scores: Path) -> dict:
    payload = json.loads(scores.read_text())
    return payload["experiments"][EXP]["results"]


def stage_outputs(dest: Path, names: list[str]) -> None:
    for name in names:
        src = PUBLISHED.get(name, CASES / f"{name}.py")
        shutil.copy(src, dest / f"{name}.py")


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
        ratios.append(a["cost"] / b["cost"])
        diffs.append(b["balancedness"] - a["balancedness"])
    n = len(diffs)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    return {
        "n_windows": n,
        "work_ratio_ref_over_variant_mean": statistics.fmean(ratios),
        "work_ratio_min": min(ratios),
        "work_ratio_max": max(ratios),
        "balance_diff_mean": statistics.fmean(diffs),
        "balance_diff_se": sd / math.sqrt(n) if n else 0.0,
        "balance_diff_min": min(diffs),
        "balance_diff_max": max(diffs),
        "windows_variant_cheaper": sum(r > 1 for r in ratios),
        "windows_variant_better_balanced": sum(d > 0 for d in diffs),
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
            "reference_in_this_run": {k: mine.get(k) for k in ("speedup_vs_frontier_reference", "balancedness", "run_failed")},
            "reference_unchanged": all(
                mine.get(k) == ref_combined.get(k) for k in ("speedup_vs_frontier_reference", "balancedness")
            ),
            **(extra or {}),
        }
        log(f"{tag}: reference unchanged = {out[tag]['reference_unchanged']}")
        save()

    # (a) slow down / inflate the published comparator policy
    if "shortcut_inflate_baseline" in out:
        pass
    else:
        _inflate(compare)
    if "shortcut_remove_frontier" not in out:
        _remove(compare)
    if "shortcut_flatten_workload" not in out:
        _flatten(compare)
    if "shortcut_forged_summary" not in out:
        _forged(compare)
    if "shortcut_omit_method_from_roster" not in out:
        _roster(compare)
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
            {"edited_frontier_speedup": res.get("frontier_reference_zigzag", {}).get("speedup_vs_frontier_reference")})


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
        for seq in payload["sequences"]:
            for w in seq["windows"]:
                w["load"] = [[1] * len(row) for row in w["load"]]
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


def mean_counts(details: dict) -> dict:
    out = {}
    for name, rec in details.items():
        if rec.get("failure") or not rec.get("calls"):
            continue
        out[name] = [statistics.fmean(c["counts"][i] for c in rec["calls"]) for i in range(5)]
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
    combined = results_of(RESULTS / "combined_scores_test.json")
    combined_payload = json.loads((RESULTS / "combined_scores_test.json").read_text())
    details = json.loads(details_test.read_text())

    details_train = RESULTS / "honest_details_train.json"
    if not stage_done("honest_train"):
        train_dir = fresh_dir(WORK / "train")
        stage_outputs(train_dir, HONEST)
        run_eval("train", train_dir, RESULTS / "honest_scores_train.json", details_train, "honest_train")
        mark_done("honest_train")
    train = results_of(RESULTS / "honest_scores_train.json")

    disk = disk_level_runs(combined)

    metrics = scored_metrics()
    honest_present = [m for m in HONEST if m in combined]
    counts_now = mean_counts(details)
    determinism = None
    job1 = RESULTS / "job1_mean_counts_test.json"
    if job1.exists():
        previous = json.loads(job1.read_text())
        determinism = {
            name: {"identical_raw_counters": [round(a, 6) for a in counts_now[name]]
                   == [round(b, 6) for b in previous[name][0]]}
            for name in counts_now if name in previous
        }
    summary = {
        "evaluator_hash": EVAL_HASH,
        "determinism_vs_job1_counters": determinism,
        "mean_counts_test": counts_now,
        "scored_metrics": metrics,
        "anchor": combined_payload.get("_anchor"),
        "failures": combined_payload.get("_failures"),
        "test": combined,
        "train_honest": train,
        "ranks_honest_cohort": cohort_ranks(combined, honest_present, metrics),
        "ranks_full_cohort": cohort_ranks(combined, list(combined), metrics),
        "paired_vs_reference": {m: paired(details, m) for m in combined if m != REFERENCE},
        "disk_level_shortcuts": disk,
    }
    ok = [m for m in combined if not combined[m]["run_failed"]]
    xs = [combined[m]["mean_rebalance_cost"] for m in ok]
    ys = [combined[m]["diagnostic_cpu_ms_per_rebalance"] for m in ok]
    summary["work_measure_calibration"] = {
        "methods": {m: {"work_units": combined[m]["mean_rebalance_cost"],
                        "cpu_ms_unmetered": combined[m]["diagnostic_cpu_ms_per_rebalance"],
                        "units_per_cpu_ms": combined[m]["mean_rebalance_cost"]
                        / max(combined[m]["diagnostic_cpu_ms_per_rebalance"], 1e-9)} for m in ok},
        "spearman_work_vs_cpu": spearman(xs, ys) if len(ok) > 2 else None,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = ["| method | speedup | balancedness | failed | rank Δ honest | rank Δ full | reason |",
             "|---|---:|---:|---:|---:|---:|---|"]
    rh = summary["ranks_honest_cohort"]["ranks"]
    rf = summary["ranks_full_cohort"]["ranks"]
    for m in all_names:
        if m not in combined:
            continue
        r = combined[m]
        reason = (summary["failures"] or {}).get(m, "")
        lines.append(
            f"| {m} | {r['speedup_vs_frontier_reference']:.4f} | {r['balancedness']:.4f} | "
            f"{int(r['run_failed'])} | {rh.get(m, {}).get('rank_delta', float('nan')):.3f} | "
            f"{rf[m]['rank_delta']:.3f} | {reason[:110]} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    log("wrote results/summary.json and results/summary.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
