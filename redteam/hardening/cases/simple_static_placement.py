# AUDIT CASE: simple load-oblivious baseline (fixed placement, no load use)
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
