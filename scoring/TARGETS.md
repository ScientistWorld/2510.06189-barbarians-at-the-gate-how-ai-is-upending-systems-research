# Evaluation Targets

## Primary: eplb_moe_placement

Reduce rebalance work relative to the frozen frontier-lab reference policy on the same windows.
- **Metric**: `speedup_vs_frontier_reference` (higher is better; coefficient 1.0). The paper reports 5.3x (19.6 ms -> 3.7 ms); the gym measures deterministic work units instead of wall-clock time.

## Constraint: eplb_moe_placement

Keep GPU load balance on the next window at least at the reference's level, with valid placements.
- **Metric**: `balancedness` (higher is better; coefficient 1.0). The paper reports an imbalance factor of 0.66 for all three policies.
- **Hard constraint**: balancedness more than 0.005 below the frontier reference, any invalid placement, a guard violation or non-determinism fails the run. A failed run scores 0 on both metrics.

Unscored diagnostics: `mean_rebalance_cost`, `balance_change_vs_frontier_reference`, `worst_window_balancedness`, `valid_fraction`, `run_failed`, `diagnostic_cpu_ms_per_rebalance`, `num_rebalances`.
