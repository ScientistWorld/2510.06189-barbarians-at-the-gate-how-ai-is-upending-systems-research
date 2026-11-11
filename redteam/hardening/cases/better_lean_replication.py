# AUDIT CASE: honest speedup -- compact log2phy and 4 kernels per replication step instead of ~8
# SPDX-License-Identifier: Apache-2.0
"""Paper's evolved EPLB policy (reproduction): zigzag reused across EPLB stages.

Method ``evolved_zigzag``.  The paper (Section 5.2, Figure 5b) reports that the
evolved program replaces loop-based greedy bin packing with a tensorised
zigzag/"snake" assignment (sort once; within even blocks of ``num_packs`` items
assign packs in order, within odd blocks in reverse, via ``torch.where``), and
that the breakthrough came from reusing this zigzag partitioning across the
EPLB stages instead of a single stage.  The replication stage keeps the rule of
repeatedly replicating the currently most overloaded expert.

Implementation: the paper's initial program (DeepSeek/vLLM EPLB) with the
tensorised zigzag used for BOTH packing stages -- expert groups onto nodes and
physical replicas onto the GPUs of each node -- batched over all layers, so no
Python loop over layers or items remains.  Everything else (hierarchical
structure, overloaded-expert replication, output maps) is unchanged.

Contract: rebalance_experts(weight[L, E], num_replicas, num_groups, num_nodes,
num_gpus) -> (phy2log[L, R], log2phy[L, E, X], logcnt[L, E]).
"""
# EVOLVE-BLOCK-START

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tensorised zigzag packing of n weighted items into m packs of n/m items.

    Items are sorted once by weight; the k-th heaviest item goes to pack
    k % num_packs in even blocks of num_packs items and to the mirrored pack
    in odd blocks, so heavy and light items alternate across packs.  No Python
    loop over layers or items.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    order = weight.float().argsort(dim=-1, descending=True, stable=True)
    ranks = torch.arange(num_items, dtype=torch.int64, device=weight.device)
    block = ranks // num_packs
    lane = ranks % num_packs
    pack_of_rank = torch.where(block % 2 == 0, lane, num_packs - 1 - lane)
    pack_index = torch.empty_like(order)
    pack_index.scatter_(-1, order, pack_of_rank.expand(num_layers, -1))
    rank_in_pack = torch.empty_like(order)
    rank_in_pack.scatter_(-1, order, block.expand(num_layers, -1))
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
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
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    maxlogcnt = int(logcnt.max().item())
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

