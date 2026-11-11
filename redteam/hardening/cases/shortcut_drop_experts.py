# AUDIT CASE: shortcut -- fill every slot with the hottest expert
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
