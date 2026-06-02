# Evaluation Targets

## Primary: eplb_moe_placement

Tests whether an expert-placement policy reduces rebalance work on changing MoE expert-load traces.
- **Metric**: `work_efficiency_vs_frontier_reference` (higher is better)
- **Diagnostic**: `work_efficiency_vs_greedy` is reported in `scores.json` to show improvement over a lightweight fixed reference, but it is not the primary target because the gym scores machine-independent operation counts rather than wall-clock runtime.

## Constraint: eplb_moe_placement

Tests whether the policy maintains balanced and valid GPU assignments rather than reducing work by ignoring placement quality.
- **Metrics**: `mean_imbalance_factor` and `valid_fraction` (higher is better)
