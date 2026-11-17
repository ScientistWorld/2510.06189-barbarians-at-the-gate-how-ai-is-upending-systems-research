# AUDIT CASE: probe -- reference algorithm in Python built from slicing, `in`, `+`/`*` on lists, str.join and C iterators
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
