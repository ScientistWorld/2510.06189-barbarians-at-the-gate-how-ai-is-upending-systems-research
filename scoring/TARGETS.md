# Evaluation Targets

## Primary: eplb_moe_placement

Tests whether an expert-placement policy reduces rebalance work on changing MoE expert-load traces.
- **Metric**: `work_efficiency_vs_greedy` (higher is better)

## Constraint: eplb_moe_placement

Tests whether the policy maintains balanced and valid GPU assignments rather than reducing work by ignoring placement quality.
- **Metrics**: `mean_imbalance_factor` and `valid_fraction` (higher is better)
