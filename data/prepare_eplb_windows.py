#!/usr/bin/env python3
"""Build the visible (train) EPLB rebalance pairs from the paper's trace.

Source: the expert-load trace released with the paper's EPLB case study
(Hugging Face dataset ``abmfy/eplb-openevolve``, file ``expert-load.json``):
1174 recorded inference steps of a 58-layer, 256-expert MoE model serving
ShareGPT and GSM8K requests.  Following the paper's evaluator, consecutive
steps are summed into windows of 100 steps; a policy rebalances on one window
and is scored on the next one.

The trace is cut into four equal time blocks.  The visible slice holds only
the *even* MoE layers (29 of 58) in blocks 0 and 2; the hidden test uses the
odd layers in blocks 1 and 3, so the visible layers never cover a time window
the hidden test is scored on, and no hidden layer is ever visible.

Each block contributes 10 rebalance pairs, starting every 10 steps: the input
window is steps ``[s, s + 100)`` and the scored window is ``[s + 100, s + 200)``.
Expert ids are relabelled with a fixed public permutation that preserves the
expert-group structure (groups are permuted, experts are permuted within their
group), the same procedure the hidden test applies with its own permutation.

Usage:
    python data/prepare_eplb_windows.py --raw data/downloads/eplb-openevolve/expert-load.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "downloads" / "eplb-openevolve" / "expert-load.json"
OUT_DEFAULT = ROOT / "data" / "eplb_train_windows.json"
RAW_SHA256 = "d93580b3cb3e404a60237fa708271b360229a0c6fdfdd924a8ec4fdb440b71ad"

INTERVAL = 100
PAIR_STRIDE = 10
PAIRS_PER_BLOCK = 10
BLOCKS = ((0, 294), (587, 880))  # time blocks 0 and 2 of the trace's four
CONFIG = {"num_replicas": 288, "num_groups": 8, "num_nodes": 4, "num_gpus": 32}
LAYERS = tuple(range(0, 58, 2))
RELABEL_SEED = 20251006


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relabelling(num_experts: int, num_groups: int, rng: random.Random) -> list[int]:
    """Return new_id for each old expert id; keeps whole groups together."""
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


def build(raw_path: Path, layers: tuple[int, ...], seed: int) -> dict:
    history = json.loads(raw_path.read_text())["load_history"]
    num_experts = len(history[0]["logical_expert_load"][0])
    rng = random.Random(seed)
    layer_order = list(layers)
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
        "description": (
            "Visible EPLB rebalance pairs: per-window summed expert loads for the even MoE "
            "layers of the paper's ShareGPT/GSM8K trace in time blocks 0 and 2 (of 4), "
            "relabelled within expert groups. Rebalance on 'load', scored on 'next_load'."
        ),
        "source": {
            "dataset": "https://huggingface.co/datasets/abmfy/eplb-openevolve",
            "file": "expert-load.json",
            "sha256": RAW_SHA256,
        },
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
    parser.add_argument("--raw", default=str(RAW_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    raw_path = Path(args.raw)
    digest = sha256(raw_path)
    if digest != RAW_SHA256:
        raise SystemExit(f"unexpected sha256 for {raw_path}: {digest}")
    payload = build(raw_path, LAYERS, RELABEL_SEED)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({payload['num_layers']} layers, {len(payload['pairs'])} rebalance pairs)")


if __name__ == "__main__":
    main()
