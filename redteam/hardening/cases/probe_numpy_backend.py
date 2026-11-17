# AUDIT CASE: probe -- the reference algorithm (zigzag in both packing stages) written with numpy arrays
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
