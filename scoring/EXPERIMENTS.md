# Experiment Overview

## eplb_moe_placement

Tests the central MoE serving question from the paper: whether a policy can adapt expert replica placement to changing expert-load traces with less rebalance work while still producing balanced, valid GPU assignments. The experiment compares candidate placement artifacts on the same deterministic workload set and reports both benefit and constraint metrics.
