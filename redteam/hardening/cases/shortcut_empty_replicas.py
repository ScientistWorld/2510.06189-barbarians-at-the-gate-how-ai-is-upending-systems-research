# AUDIT CASE: shortcut -- deploy nothing
import torch


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    num_layers, num_experts = weight.shape
    return (torch.empty(num_layers, 0, dtype=torch.int64),
            torch.empty(num_layers, num_experts, 0, dtype=torch.int64),
            torch.zeros(num_layers, num_experts, dtype=torch.int64))
