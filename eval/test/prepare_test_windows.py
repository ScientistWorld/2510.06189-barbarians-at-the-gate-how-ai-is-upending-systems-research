#!/usr/bin/env python3
"""Regenerate the hidden EPLB test pairs (held-out layers, held-out time).

The committed ``eval/test/test_windows.json`` is what the hidden test reads;
this script documents how it was produced and can rebuild it from the paper's
public trace.  The trace is cut into four equal time blocks.  The hidden test
uses only the *odd* MoE layers in blocks 1 and 3: no visible layer covers
these time windows, and no hidden layer is visible at any time.  Layers are
shuffled and expert ids relabelled (groups permuted, experts permuted within
their group) with a seed that lives only in this directory.

Each block contributes 10 rebalance pairs, starting every 10 steps: the input
window is steps ``[s, s + 100)`` and the scored window is ``[s + 100, s + 200)``.

Usage:
    python eval/test/prepare_test_windows.py --raw /path/to/expert-load.json
    python eval/test/prepare_test_windows.py --download   # fetch into a temp dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "test_windows.json"
RAW_URL = (
    "https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/"
    "071910dada7ce35e8b9b89b64ec1295f1b64590b/expert-load.json"
)
RAW_SHA256 = "d93580b3cb3e404a60237fa708271b360229a0c6fdfdd924a8ec4fdb440b71ad"

INTERVAL = 100
PAIR_STRIDE = 10
PAIRS_PER_BLOCK = 10
BLOCKS = ((294, 587), (880, 1174))  # time blocks 1 and 3 of the trace's four
CONFIG = {"num_replicas": 288, "num_groups": 8, "num_nodes": 4, "num_gpus": 32}
LAYERS = tuple(range(1, 58, 2))
SECRET_SEED = 7_391_442_017


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relabelling(num_experts: int, num_groups: int, rng: random.Random) -> list[int]:
    group_size = num_experts // num_groups
    group_perm = list(range(num_groups))
    rng.shuffle(group_perm)
    new_id = [0] * num_experts
    for old_group in range(num_groups):
        local = list(range(group_size))
        rng.shuffle(local)
        for old_local in range(group_size):
            new_id[old_group * group_size + old_local] = (
                group_perm[old_group] * group_size + local[old_local]
            )
    return new_id


def window(history, start, layer_order, maps, num_experts):
    out = []
    for layer in layer_order:
        summed = [0] * num_experts
        for entry in history[start:start + INTERVAL]:
            row = entry["logical_expert_load"][layer]
            for expert, value in enumerate(row):
                summed[expert] += int(value)
        relabelled = [0] * num_experts
        for old, new in enumerate(maps[layer]):
            relabelled[new] = summed[old]
        out.append(relabelled)
    return out


def build(raw_path: Path) -> dict:
    history = json.loads(raw_path.read_text())["load_history"]
    num_experts = len(history[0]["logical_expert_load"][0])
    rng = random.Random(SECRET_SEED)
    layer_order = list(LAYERS)
    rng.shuffle(layer_order)
    maps = {layer: relabelling(num_experts, CONFIG["num_groups"], rng) for layer in layer_order}

    pairs = []
    for a, b in BLOCKS:
        for k in range(PAIRS_PER_BLOCK):
            start = a + k * PAIR_STRIDE
            assert start + 2 * INTERVAL <= b <= len(history)
            pairs.append({
                "block": [a, b],
                "start": start,
                "load": window(history, start, layer_order, maps, num_experts),
                "next_load": window(history, start + INTERVAL, layer_order, maps, num_experts),
            })
    return {
        "description": "Hidden EPLB test pairs (held-out MoE layers in held-out time blocks, secretly relabelled).",
        "interval_steps": INTERVAL,
        "pair_stride_steps": PAIR_STRIDE,
        "blocks": [list(block) for block in BLOCKS],
        "config": dict(CONFIG),
        "num_layers": len(layer_order),
        "num_logical_experts": num_experts,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.raw:
            raw_path = Path(args.raw)
        elif args.download:
            raw_path = Path(tmp) / "expert-load.json"
            subprocess.run(["curl", "-sSL", "-o", str(raw_path), RAW_URL], check=True)
        else:
            raise SystemExit("pass --raw PATH or --download")
        digest = sha256(raw_path)
        if digest != RAW_SHA256:
            raise SystemExit(f"unexpected sha256 for {raw_path}: {digest}")
        payload = build(raw_path)

    out = Path(args.output)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({payload['num_layers']} layers, {len(payload['pairs'])} rebalance pairs)")


if __name__ == "__main__":
    main()
