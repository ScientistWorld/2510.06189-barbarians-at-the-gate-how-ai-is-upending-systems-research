#!/usr/bin/env python3
"""Regenerate the hidden EPLB test windows (held-out MoE layers).

The committed ``eval/test/test_windows.json`` is what the hidden test reads;
this script documents how it was produced and can rebuild it from the paper's
public trace.  It emits only the *odd* MoE layers, with a layer shuffle and an
expert relabelling (groups permuted, experts permuted within their group) drawn
from a seed that lives only in this directory.

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
OFFSETS = (0, 25, 50, 75)
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


def build(raw_path: Path) -> dict:
    history = json.loads(raw_path.read_text())["load_history"]
    num_experts = len(history[0]["logical_expert_load"][0])
    rng = random.Random(SECRET_SEED)
    layer_order = list(LAYERS)
    rng.shuffle(layer_order)
    maps = {layer: relabelling(num_experts, CONFIG["num_groups"], rng) for layer in layer_order}

    sequences = []
    for offset in OFFSETS:
        windows = []
        for start in range(offset, len(history), INTERVAL):
            end = min(start + INTERVAL, len(history))
            window = []
            for layer in layer_order:
                summed = [0] * num_experts
                for entry in history[start:end]:
                    row = entry["logical_expert_load"][layer]
                    for expert, value in enumerate(row):
                        summed[expert] += int(value)
                relabelled = [0] * num_experts
                for old, new in enumerate(maps[layer]):
                    relabelled[new] = summed[old]
                window.append(relabelled)
            windows.append({"start": start, "end": end, "load": window})
        sequences.append({"offset": offset, "windows": windows})

    return {
        "description": "Hidden EPLB test windows (held-out MoE layers, secretly relabelled).",
        "interval_steps": INTERVAL,
        "config": dict(CONFIG),
        "num_layers": len(layer_order),
        "num_logical_experts": num_experts,
        "sequences": sequences,
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
    n_windows = sum(len(s["windows"]) for s in payload["sequences"])
    print(f"wrote {out} ({payload['num_layers']} layers, {n_windows} windows)")


if __name__ == "__main__":
    main()
