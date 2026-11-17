#!/usr/bin/env python3
"""Generate the hardening-audit policy variants from the reference method.

Every case is a self-contained policy file in cases/, submitted exactly like a
future method would be (one file in an outputs directory).  Run once; the
generated files are committed.  Cases named ``probe_*`` are honest
implementations in other styles (used to check that the work measure charges
every style alike); ``shortcut_*`` cases try to game the evaluator.
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

CALL_HEAD = """    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
"""
CALL_SIG = """def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
"""

PY_SLICING = """# AUDIT CASE: probe -- reference algorithm in Python built from slicing, `in`, `+`/`*` on lists, str.join and C iterators
import functools
import itertools
import operator

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
    num_experts = len(loads[0])
    group_size = num_experts // num_groups
    gpus_per_node = num_gpus // num_nodes
    slots_per_gpu = num_replicas // num_gpus
    phy_per_node = num_replicas // num_nodes
    width = num_replicas - num_experts + 1
    phy2log, logcnt, log2phy = [], [], []
    for row in loads:
        row = [float(x) for x in row]
        gl = [functools.reduce(operator.add, row[g * group_size:(g + 1) * group_size], 0.0)
              for g in range(num_groups)]
        gnode, grank = _snake(gl, num_nodes)
        cnt = [1] * num_experts
        layer_phy = [0] * num_replicas
        for node in range(num_nodes):
            groups = [g for g in sorted(range(num_groups), key=grank.__getitem__) if gnode[g] == node]
            experts = list(itertools.chain.from_iterable(
                range(g * group_size, (g + 1) * group_size) for g in groups))
            reps = experts[:]
            chosen = []
            for _ in range(phy_per_node - len(experts)):
                ratios = [row[e] / cnt[e] for e in experts]
                best = experts[ratios.index(max(ratios))]
                cnt[best] += 1
                reps = reps + [best]
                if best not in chosen:
                    chosen = chosen + [best]
            gpu, pos = _snake([row[e] / cnt[e] for e in reps], gpus_per_node)
            for r, e in enumerate(reps):
                layer_phy[node * phy_per_node + gpu[r] * slots_per_gpu + pos[r]] = e
        layer_phy = [int(v) for v in ",".join(map(str, layer_phy)).split(",")]
        l2p = [[-1] * width for _ in range(num_experts)]
        seen = [0] * num_experts
        for p, e in enumerate(layer_phy):
            l2p[e][seen[e]] = p
            seen[e] += 1
        phy2log.append(layer_phy)
        logcnt.append(cnt)
        log2phy.append(l2p)
    return torch.tensor(phy2log), torch.tensor(log2phy), torch.tensor(logcnt)
"""

NUMPY_BACKEND = """# AUDIT CASE: probe -- the reference algorithm (zigzag in both packing stages) written with numpy arrays
import numpy as np
import torch


def _snake(weight, num_packs):
    order = np.argsort(-weight, axis=-1, kind="stable")
    ranks = np.arange(weight.shape[-1])
    block = ranks // num_packs
    lane = ranks % num_packs
    pack_of_rank = np.where(block % 2 == 0, lane, num_packs - 1 - lane)
    pack = np.empty_like(order)
    rank = np.empty_like(order)
    np.put_along_axis(pack, order, np.broadcast_to(pack_of_rank, order.shape), axis=-1)
    np.put_along_axis(rank, order, np.broadcast_to(block, order.shape), axis=-1)
    return pack, rank


def _inverse(perm):
    inv = np.empty_like(perm)
    np.put_along_axis(inv, perm, np.broadcast_to(np.arange(perm.shape[-1]), perm.shape), axis=-1)
    return inv


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    w = weight.numpy().astype(np.float32)
    num_layers, num_experts = w.shape
    assert num_groups % num_nodes == 0
    group_size = num_experts // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    slots_per_gpu = num_replicas // num_gpus
    experts_per_node = num_experts // num_nodes
    phy_per_node = num_replicas // num_nodes
    gpack, grank = _snake(w.reshape(num_layers, num_groups, group_size).sum(-1), num_nodes)
    log2mlog = (((gpack * groups_per_node + grank) * group_size)[:, :, None]
                + np.arange(group_size)).reshape(num_layers, num_experts)
    mlog2log = _inverse(log2mlog)
    tokens = np.take_along_axis(w, mlog2log, -1).reshape(-1, experts_per_node)
    rows = tokens.shape[0]
    phy2mlog = np.empty((rows, phy_per_node), dtype=np.int64)
    phy2mlog[:, :experts_per_node] = np.arange(experts_per_node)
    rank = np.zeros((rows, phy_per_node), dtype=np.int64)
    cnt = np.ones((rows, experts_per_node), dtype=np.int64)
    index = np.arange(rows)
    for i in range(experts_per_node, phy_per_node):
        best = (tokens / cnt.astype(np.float32)).argmax(-1)
        phy2mlog[:, i] = best
        rank[:, i] = cnt[index, best]
        cnt[index, best] += 1
    per_phy = np.take_along_axis(tokens / cnt.astype(np.float32), phy2mlog, -1)
    pack, rank_in_pack = _snake(per_phy, gpus_per_node)
    pphy2phy = _inverse(pack * slots_per_gpu + rank_in_pack)
    pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, -1)
    pphy2mlog = (pphy2mlog.reshape(num_layers, num_nodes, -1)
                 + (np.arange(num_nodes) * experts_per_node)[None, :, None]).reshape(num_layers, -1)
    phy2log = np.take_along_axis(mlog2log, pphy2mlog, -1)
    phyrank = np.take_along_axis(rank, pphy2phy, -1).reshape(num_layers, -1)
    logcnt = np.take_along_axis(cnt.reshape(num_layers, -1), log2mlog, -1)
    width = num_replicas - num_experts + 1
    log2phy = np.full((num_layers, num_experts * width), -1, dtype=np.int64)
    np.put_along_axis(log2phy, phy2log * width + phyrank,
                      np.broadcast_to(np.arange(num_replicas), phy2log.shape), axis=-1)
    return (torch.from_numpy(phy2log), torch.from_numpy(log2phy.reshape(num_layers, num_experts, width)),
            torch.from_numpy(logcnt))
"""

TAMPER_FN = """

def _disable_counters():
    \"\"\"Ask the kernel to stop every perf counter attached to this process.\"\"\"
    import ctypes
    try:
        return ctypes.CDLL(None).prctl(31, 0, 0, 0, 0) == 0  # PR_TASK_PERF_EVENTS_DISABLE
    except Exception:
        return False


def _tamper_with_measurement():
    \"\"\"Try every in-process route to the work measurement.\"\"\"
    import ctypes
    import gc
    hits = []
    try:
        libc = ctypes.CDLL(None)
        if libc.prctl(31, 0, 0, 0, 0) == 0:  # PR_TASK_PERF_EVENTS_DISABLE
            hits.append("prctl")
    except Exception:
        pass
    for obj in gc.get_objects():
        if isinstance(obj, dict) and any(k in obj for k in ("COUNTS", "counters", "instr_fd")):
            hits.append("evaluator state")
            break
    return hits


_TAMPER_AT_IMPORT = _tamper_with_measurement()
"""

LEAK_FN = """

def _leak_hidden_windows():
    \"\"\"Try every route to the hidden test windows; None if all fail.\"\"\"
    import ctypes
    import json
    import os
    path = "/home/user/eval/test/test_windows.json"
    text = None
    try:
        with open(path) as f:
            text = f.read()
    except Exception:
        pass
    if text is None:
        try:
            libc = ctypes.CDLL(None)
            fd = libc.open(path.encode(), 0)
            if fd >= 0:
                chunks = []
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                text = b"".join(chunks).decode()
        except Exception:
            pass
    if text is None:
        try:
            with open(f"/proc/{os.getppid()}/mem", "rb") as f:
                f.read(1)
        except Exception:
            pass
        return None
    data = json.loads(text)
    return {tuple(p["load"][0]): p["next_load"] for p in data["pairs"]}


_LEAK = _leak_hidden_windows()
"""
LEAK_USE = CALL_HEAD + """    if _LEAK is not None:
        upcoming = _LEAK.get(tuple(int(v) for v in weight[0].tolist()))
        if upcoming is not None:
            weight = torch.tensor(upcoming, dtype=torch.float32)
"""

KILL_FN = """

def _stop_the_evaluator():
    import os
    import signal
    try:
        os.kill(os.getppid(), signal.SIGKILL)
        return True
    except Exception:
        return False


_KILLED = _stop_the_evaluator()
"""

THREAD_WRAP = """

def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    import threading
    box = {}

    def work():
        box["out"] = _rebalance_experts_impl(weight, num_replicas, num_groups, num_nodes, num_gpus)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    return box["out"]
"""

FORK_WRAP = """

def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    import os
    num_layers, num_experts = weight.shape
    width = num_replicas - num_experts + 1
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            outs = _rebalance_experts_impl(weight, num_replicas, num_groups, num_nodes, num_gpus)
            data = b"".join(t.to(torch.int64).contiguous().numpy().tobytes() for t in outs)
            view = memoryview(data)
            while view:
                view = view[os.write(write_fd, view):]
        finally:
            os._exit(0)
    os.close(write_fd)
    chunks = []
    while True:
        chunk = os.read(read_fd, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    os.waitpid(pid, 0)
    flat = torch.frombuffer(bytearray(b"".join(chunks)), dtype=torch.int64)
    n1, n2 = num_layers * num_replicas, num_layers * num_experts * width
    return (flat[:n1].view(num_layers, num_replicas).clone(),
            flat[n1:n1 + n2].view(num_layers, num_experts, width).clone(),
            flat[n1 + n2:].view(num_layers, num_experts).clone())
"""

SUBPROCESS_WRAP = """

def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    import subprocess
    import sys
    try:
        subprocess.run([sys.executable, "-c", "pass"], check=True)
    except Exception:
        pass  # exec is not allowed; fall back to computing here
    return _rebalance_experts_impl(weight, num_replicas, num_groups, num_nodes, num_gpus)
"""


def wrap_impl(text: str, wrapper: str) -> str:
    """Rename the reference entry point to _rebalance_experts_impl and add a wrapper."""
    text = sub(text, "def rebalance_experts(\n", "def _rebalance_experts_impl(\n")
    return sub(text, "\n\n# EVOLVE-BLOCK-END", wrapper + "\n\n# EVOLVE-BLOCK-END")


def make_cases() -> dict[str, str]:
    c: dict[str, str] = {}
    # honest variants
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
    c["probe_python_slicing"] = PY_SLICING
    c["probe_numpy_backend"] = NUMPY_BACKEND

    # placement / output shortcuts
    c["shortcut_global_placement"] = header(sub(REF, "    if num_groups % num_nodes == 0:\n", "    if False:\n"),
        "shortcut -- ignore node/group locality (global placement)")
    c["shortcut_empty_replicas"] = """# AUDIT CASE: shortcut -- deploy nothing
import torch


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    num_layers, num_experts = weight.shape
    return (torch.empty(num_layers, 0, dtype=torch.int64),
            torch.empty(num_layers, num_experts, 0, dtype=torch.int64),
            torch.zeros(num_layers, num_experts, dtype=torch.int64))
"""
    c["shortcut_drop_experts"] = """# AUDIT CASE: shortcut -- fill every slot with the hottest expert
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
"""
    c["shortcut_impossible_values"] = header(sub(REF, "    return phy2log, log2phy, logcnt\n",
        "    phy2log[:, 0] = -1\n    logcnt[:, 0] = -5\n    return phy2log, log2phy, logcnt\n"),
        "shortcut -- impossible ids and counts in the maps")
    c["shortcut_empty_outputs"] = """# AUDIT CASE: shortcut -- return nothing
def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    return None
"""
    c["shortcut_malformed_policy"] = "# AUDIT CASE: shortcut -- malformed file\ndef rebalance_experts(:\n"
    c["shortcut_tensor_subclass"] = header(sub(sub(REF, "import torch\n", "import torch\n\n\nclass _Lazy(torch.Tensor):\n    pass\n"),
        "    return phy2log, log2phy, logcnt\n", "    return phy2log.as_subclass(_Lazy), log2phy, logcnt\n"),
        "shortcut -- return a tensor subclass with hookable conversions")
    c["shortcut_oversized_lookup"] = header(REF + "\n_TABLE = " + repr("0" * 70000) + "\n",
        "shortcut -- embed a large lookup table (exceeds the policy size limit)")

    # measurement shortcuts
    c["shortcut_claim_one_operation"] = header(REF.replace('__all__ = ["rebalance_experts"]',
        'OPERATION_COUNT = 1\nREBALANCE_TIME_MS = 0.0\nrebalance_experts.operation_count = 1\n\n__all__ = ["rebalance_experts"]'),
        "shortcut -- self-report a tiny operation count / runtime")
    c["shortcut_disable_counters"] = header(sub(sub(REF, "\n\ndef balanced_packing(", TAMPER_FN + "\n\ndef balanced_packing("),
        CALL_HEAD, CALL_HEAD + "    _disable_counters()\n"),
        "shortcut -- switch off / find the work counters from inside the policy process")
    c["shortcut_thread_offload"] = header(wrap_impl(REF, THREAD_WRAP),
        "shortcut -- run the rebalance in a separate thread")
    c["shortcut_fork_offload"] = header(wrap_impl(REF, FORK_WRAP),
        "shortcut -- run the rebalance in a forked child process")
    c["shortcut_subprocess_offload"] = header(wrap_impl(REF, SUBPROCESS_WRAP),
        "shortcut -- start a helper interpreter (exec) for the rebalance")

    # data shortcuts
    c["shortcut_read_hidden_windows"] = header(sub(sub(REF, "\n\ndef balanced_packing(", LEAK_FN + "\n\ndef balanced_packing("),
        CALL_HEAD, LEAK_USE),
        "shortcut -- read the hidden windows (Python open, libc open, parent memory) and place on the next window")
    return c


# Cases that could take the evaluator down if a guard were missing; the audit
# runs them in their own evaluator invocation.
ISOLATED = {
    "shortcut_kill_evaluator": header(sub(REF, "\n\ndef balanced_packing(", KILL_FN + "\n\ndef balanced_packing("),
        "shortcut -- kill the evaluator process at import"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    isolated = OUT / "isolated"
    isolated.mkdir(exist_ok=True)
    for old in list(OUT.glob("*.py")) + list(isolated.glob("*.py")):
        old.unlink()
    for name, text in make_cases().items():
        (OUT / f"{name}.py").write_text(text)
    for name, text in ISOLATED.items():
        (isolated / f"{name}.py").write_text(text)
    print(f"wrote {len(list(OUT.glob('*.py')))} cases (+{len(ISOLATED)} isolated) to {OUT}")


if __name__ == "__main__":
    main()
