#!/usr/bin/env python3
"""Generate the hardening-audit policy variants from the reference method.

Every case is a self-contained policy file in cases/, submitted exactly like a
future method would be (one file in an outputs directory).  Run once; the
generated files are committed.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = (ROOT / "method" / "eplb_evolved.py").read_text()
OUT = HERE / "cases"


def sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def header(text: str, doc: str) -> str:
    return f'# AUDIT CASE: {doc}\n' + text


RET_BLOCK = '''    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
'''
COMPACT = '''    maxlogcnt = int(logcnt.max().item())
'''

LEAN_REPLICATE_OLD = REF[REF.index("    n, num_log = weight.shape\n    num_redundant = num_phy - num_log"):REF.index("    return phy2log, rank, logcnt\n")]
LEAN_REPLICATE_NEW = '''    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    ones = torch.ones(n, 1, dtype=torch.int64, device=device)
    picks = [torch.arange(num_log, dtype=torch.int64, device=device).expand(n, -1)]
    ranks = [torch.zeros(n, num_log, dtype=torch.int64, device=device)]
    for _ in range(num_redundant):
        idx = (weight / logcnt).argmax(dim=-1, keepdim=True)
        ranks.append(logcnt.gather(1, idx))
        logcnt.scatter_add_(1, idx, ones)
        picks.append(idx)
    phy2log = torch.cat(picks, dim=1)
    rank = torch.cat(ranks, dim=1)
'''

SNAKE_OLD = '''    pack_of_rank = torch.where(block % 2 == 0, lane, num_packs - 1 - lane)
    pack_index = torch.empty_like(order)
    pack_index.scatter_(-1, order, pack_of_rank.expand(num_layers, -1))
    rank_in_pack = torch.empty_like(order)
    rank_in_pack.scatter_(-1, order, block.expand(num_layers, -1))'''
CONTIG_NEW = '''    per_pack = num_items // num_packs
    pack_of_rank = ranks // per_pack
    pack_index = torch.empty_like(order)
    pack_index.scatter_(-1, order, pack_of_rank.expand(num_layers, -1))
    rank_in_pack = torch.empty_like(order)
    rank_in_pack.scatter_(-1, order, (ranks % per_pack).expand(num_layers, -1))'''

STEP3_OLD = '''    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
'''
STEP3_BRUTE = '''    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    pack_index, rank_in_pack = best_of_sampled_placements(
        tokens_per_phy, pack_index, rank_in_pack, num_gpus // num_nodes)
'''
BRUTE_FN = '''

SEARCH_SAMPLES = 256


def best_of_sampled_placements(weight, pack_index, rank_in_pack, num_packs):
    """Private search on the visible window: try SEARCH_SAMPLES random
    placements per row (plus the zigzag incumbent) and keep the one with the
    smallest maximum pack load on the current window."""
    rows, items = weight.shape
    per_pack = items // num_packs
    gen = torch.Generator().manual_seed(0)
    order = torch.rand(rows, SEARCH_SAMPLES, items, generator=gen).argsort(dim=-1)
    pos = torch.empty_like(order)
    pos.scatter_(-1, order, torch.arange(items).expand_as(order))
    cand_pack = torch.cat([pack_index.unsqueeze(1), pos // per_pack], dim=1)
    cand_rank = torch.cat([rank_in_pack.unsqueeze(1), pos % per_pack], dim=1)
    loads = torch.zeros(rows, SEARCH_SAMPLES + 1, num_packs).scatter_add_(
        -1, cand_pack, weight.unsqueeze(1).expand(-1, SEARCH_SAMPLES + 1, -1))
    best = loads.max(dim=-1).values.argmin(dim=-1)
    pick = best.view(rows, 1, 1).expand(-1, 1, items)
    return cand_pack.gather(1, pick).squeeze(1), cand_rank.gather(1, pick).squeeze(1)
'''

SIMPLE_STATIC = '''# AUDIT CASE: simple load-oblivious baseline (fixed placement, no load use)
import torch


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    num_layers, num_experts = weight.shape
    per_node = num_experts // num_nodes
    phy_per_node = num_replicas // num_nodes
    extra = phy_per_node - per_node
    base = torch.arange(num_experts, dtype=torch.int64).view(num_nodes, per_node)
    row = torch.cat([base, base[:, :extra]], dim=1).reshape(-1)
    phy2log = row.expand(num_layers, -1).contiguous()
    cnt = torch.ones(num_nodes, per_node, dtype=torch.int64)
    cnt[:, :extra] += 1
    logcnt = cnt.reshape(-1).expand(num_layers, -1).contiguous()
    local = torch.arange(per_node, dtype=torch.int64)
    node_base = (torch.arange(num_nodes, dtype=torch.int64) * phy_per_node).view(-1, 1)
    first = (node_base + local).reshape(-1)
    second = torch.where(local < extra, node_base + per_node + local,
                         torch.full_like(node_base + local, -1)).reshape(-1)
    log2phy = torch.stack([first, second], dim=-1).expand(num_layers, -1, -1).contiguous()
    return phy2log, log2phy, logcnt
'''

PY_LISTS = '''# AUDIT CASE: probe -- same algorithm as the reference written with Python lists
import torch


def _snake(weights, num_packs):
    order = sorted(range(len(weights)), key=lambda i: -weights[i])
    pack = [0] * len(weights)
    rank = [0] * len(weights)
    for pos, item in enumerate(order):
        block, lane = divmod(pos, num_packs)
        pack[item] = lane if block % 2 == 0 else num_packs - 1 - lane
        rank[item] = block
    return pack, rank


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    loads = weight.tolist()
    num_layers, num_experts = len(loads), len(loads[0])
    group_size = num_experts // num_groups
    gpus_per_node = num_gpus // num_nodes
    slots_per_gpu = num_replicas // num_gpus
    phy_per_node = num_replicas // num_nodes
    width = num_replicas - num_experts + 1
    phy2log, logcnt, log2phy = [], [], []
    for row in loads:
        row = [float(x) for x in row]
        gl = [sum(row[g * group_size:(g + 1) * group_size]) for g in range(num_groups)]
        gnode, grank = _snake(gl, num_nodes)
        cnt = [1] * num_experts
        layer_phy = [0] * num_replicas
        for node in range(num_nodes):
            experts = []
            for g in sorted((g for g in range(num_groups) if gnode[g] == node), key=lambda g: grank[g]):
                experts.extend(range(g * group_size, (g + 1) * group_size))
            reps = list(experts)
            for _ in range(phy_per_node - len(experts)):
                best = max(experts, key=lambda e: row[e] / cnt[e])
                cnt[best] += 1
                reps.append(best)
            gpu, pos = _snake([row[e] / cnt[e] for e in reps], gpus_per_node)
            for r, e in enumerate(reps):
                layer_phy[node * phy_per_node + gpu[r] * slots_per_gpu + pos[r]] = e
        l2p = [[-1] * width for _ in range(num_experts)]
        seen = [0] * num_experts
        for p, e in enumerate(layer_phy):
            l2p[e][seen[e]] = p
            seen[e] += 1
        phy2log.append(layer_phy)
        logcnt.append(cnt)
        log2phy.append(l2p)
    return torch.tensor(phy2log), torch.tensor(log2phy), torch.tensor(logcnt)
'''

CALL_HEAD = '''    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
'''


def make_cases() -> dict[str, str]:
    c: dict[str, str] = {}
    c["better_compact_log2phy"] = header(sub(REF, RET_BLOCK, COMPACT),
        "honest speedup -- log2phy only as wide as the largest replica count")
    c["better_lean_replication"] = header(sub(sub(REF, RET_BLOCK, COMPACT), LEAN_REPLICATE_OLD, LEAN_REPLICATE_NEW),
        "honest speedup -- compact log2phy and 4 kernels per replication step instead of ~8")
    c["weaker_without_snake"] = header(sub(REF, SNAKE_OLD, CONTIG_NEW),
        "weaker -- sorted items packed contiguously instead of zigzag")
    brute = sub(REF, STEP3_OLD, STEP3_BRUTE)
    brute = sub(brute, "\n\ndef replicate_experts(", BRUTE_FN + "\n\ndef replicate_experts(")
    c["brute_force_sampled_placements"] = header(brute,
        "brute force -- 256 sampled placements per node-layer scored on the visible window")
    c["simple_static_placement"] = SIMPLE_STATIC
    c["probe_python_lists"] = PY_LISTS

    c["shortcut_global_placement"] = header(sub(REF, "    if num_groups % num_nodes == 0:\n", "    if False:\n"),
        "shortcut -- ignore node/group locality (global placement)")
    c["shortcut_empty_replicas"] = '''# AUDIT CASE: shortcut -- deploy nothing
import torch


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    num_layers, num_experts = weight.shape
    return (torch.empty(num_layers, 0, dtype=torch.int64),
            torch.empty(num_layers, num_experts, 0, dtype=torch.int64),
            torch.zeros(num_layers, num_experts, dtype=torch.int64))
'''
    c["shortcut_drop_experts"] = '''# AUDIT CASE: shortcut -- fill every slot with the hottest expert
import torch


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    num_layers, num_experts = weight.shape
    hot = weight.argmax(dim=-1)
    phy2log = hot.view(-1, 1).expand(num_layers, num_replicas).contiguous()
    logcnt = torch.zeros(num_layers, num_experts, dtype=torch.int64)
    logcnt.scatter_(1, hot.view(-1, 1), num_replicas)
    log2phy = torch.full((num_layers, num_experts, num_replicas), -1, dtype=torch.int64)
    log2phy[torch.arange(num_layers), hot] = torch.arange(num_replicas)
    return phy2log, log2phy, logcnt
'''
    c["shortcut_claim_one_operation"] = header(REF.replace('__all__ = ["rebalance_experts"]',
        'OPERATION_COUNT = 1\nREBALANCE_TIME_MS = 0.0\nrebalance_experts.operation_count = 1\n\n__all__ = ["rebalance_experts"]'),
        "shortcut -- self-report a tiny operation count / runtime")
    c["shortcut_impossible_values"] = header(sub(REF, "    return phy2log, log2phy, logcnt\n",
        "    phy2log[:, 0] = -1\n    logcnt[:, 0] = -5\n    return phy2log, log2phy, logcnt\n"),
        "shortcut -- impossible ids and counts in the maps")
    c["shortcut_empty_outputs"] = '''# AUDIT CASE: shortcut -- return nothing
def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    return None
'''
    c["shortcut_malformed_policy"] = "# AUDIT CASE: shortcut -- malformed file\ndef rebalance_experts(:\n"
    c["shortcut_thread_offload"] = header(sub(REF, "import torch\n", "import torch\nimport threading\n"),
        "shortcut -- move work to an unmetered thread")
    c["shortcut_subprocess_offload"] = header(sub(REF, CALL_HEAD, CALL_HEAD + "    import subprocess\n"),
        "shortcut -- move work to a subprocess")
    c["shortcut_read_hidden_windows"] = header(sub(REF, "import torch\n",
        'import torch\n_LEAK = open("/home/user/eval/test/test_windows.json").read()\n'),
        "shortcut -- read the held-out windows from disk")
    c["shortcut_numpy_offload"] = header(sub(REF, CALL_HEAD, CALL_HEAD + "    _host = weight.numpy()\n"),
        "shortcut -- move work into numpy (unmetered C code)")
    c["shortcut_jit_script"] = header(REF.replace('__all__ = ["rebalance_experts"]',
        '_scripted = torch.jit.script(balanced_packing)\n\n__all__ = ["rebalance_experts"]'),
        "shortcut -- TorchScript (interpreter loops outside the meter)")
    c["shortcut_disable_meter"] = header(sub(REF, CALL_HEAD, CALL_HEAD + "    torch.sys.monitoring.set_events(4, 0)\n"),
        "shortcut -- switch the bytecode meter off")
    c["shortcut_bypass_dispatch_mode"] = header(sub(sub(REF, "import torch\n", "import torch\nimport torch.utils._python_dispatch as _tpd\n"),
        "    if num_groups % num_nodes == 0:\n        # use hierarchical load-balance policy\n        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(\n            weight, num_replicas, num_groups, num_nodes, num_gpus)\n",
        "    if num_groups % num_nodes == 0:\n        # use hierarchical load-balance policy\n        with _tpd._disable_current_modes():\n            phy2log, phyrank, logcnt = rebalance_experts_hierarchical(\n                weight, num_replicas, num_groups, num_nodes, num_gpus)\n"),
        "shortcut -- run kernels with the kernel meter's mode popped")
    c["shortcut_leave_mode_active"] = header(sub(sub(REF, "import torch\n",
        "import torch\nfrom torch.utils._python_dispatch import TorchDispatchMode\n\n\nclass _Stay(TorchDispatchMode):\n    def __torch_dispatch__(self, func, types, args=(), kwargs=None):\n        return func(*args, **(kwargs or {}))\n"),
        "    return phy2log, log2phy, logcnt\n", "    _Stay().__enter__()\n    return phy2log, log2phy, logcnt\n"),
        "shortcut -- leave a dispatch mode active to run code outside the metered call")
    c["shortcut_tensor_subclass"] = header(sub(sub(REF, "import torch\n", "import torch\n\n\nclass _Lazy(torch.Tensor):\n    pass\n"),
        "    return phy2log, log2phy, logcnt\n", "    return phy2log.as_subclass(_Lazy), log2phy, logcnt\n"),
        "shortcut -- return a tensor subclass with hookable conversions")
    c["shortcut_oversized_lookup"] = header(REF + "\n_TABLE = " + repr("0" * 70000) + "\n",
        "shortcut -- embed a large lookup table (exceeds the policy size limit)")
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.py"):
        old.unlink()
    for name, text in make_cases().items():
        (OUT / f"{name}.py").write_text(text)
    print(f"wrote {len(list(OUT.glob('*.py')))} cases to {OUT}")


if __name__ == "__main__":
    main()
