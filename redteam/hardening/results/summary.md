| method | speedup | balancedness | instructions | cpu ms | failed | rank Δ honest | rank Δ full | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| evolved_zigzag | 4.6741 | 0.6863 | 7.229e+06 | 3.36 | 0 | 0.000 | 0.000 |  |
| frontier_reference_zigzag | 0.9992 | 0.6863 | 3.382e+07 | 7.00 | 0 | -1.716 | -1.678 |  |
| greedy_deepseek | 0.0250 | 0.6869 | 1.352e+09 | 168.70 | 0 | -2.169 | -2.121 |  |
| better_compact_log2phy | 4.7639 | 0.6863 | 7.093e+06 | 2.37 | 0 | 0.042 | 0.041 |  |
| better_lean_replication | 5.0783 | 0.6863 | 6.654e+06 | 2.44 | 0 | 0.189 | 0.185 |  |
| brute_force_sampled_placements | 0.0624 | 0.6860 | 5.417e+08 | 103.95 | 0 | -2.154 | -2.107 |  |
| probe_numpy_backend | 3.6159 | 0.6863 | 9.345e+06 | 3.09 | 0 | -0.494 | -0.483 |  |
| probe_python_lists | 0.0539 | 0.6863 | 6.274e+08 | 43.73 | 0 | -2.157 | -2.110 |  |
| probe_python_slicing | 0.0531 | 0.6863 | 6.367e+08 | 44.21 | 0 | -2.158 | -2.110 |  |
| shortcut_claim_one_operation | 4.6606 | 0.6863 | 7.25e+06 | 3.58 | 0 | nan | -0.006 |  |
| shortcut_disable_counters | 4.5931 | 0.6863 | 7.357e+06 | 3.58 | 0 | nan | -0.037 |  |
| shortcut_drop_experts | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | pair 0 (start 294): invalid placement: layer 0: logical expert 0 has no replica |
| shortcut_empty_outputs | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | pair 0 (start 294): invalid output: rebalance_experts must return (phy2log, log2phy, logcnt) |
| shortcut_empty_replicas | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | pair 0 (start 294): invalid output: phy2log has shape (29, 0): expected (29, 288) |
| shortcut_fork_offload | 3.8167 | 0.6863 | 8.853e+06 | 21.46 | 0 | nan | -0.392 |  |
| shortcut_global_placement | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | pair 0 (start 294): invalid placement: layer 0: expert group 2 is split across nodes |
| shortcut_impossible_values | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | pair 0 (start 294): invalid placement: layer 0: invalid logical expert id -1 |
| shortcut_malformed_policy | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | load failed: SyntaxError: invalid syntax (shortcut_malformed_policy.py, line 2) |
| shortcut_oversized_lookup | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | policy file is 79047 bytes (limit 65536) |
| shortcut_read_hidden_windows | 4.6722 | 0.6863 | 7.232e+06 | 3.49 | 0 | nan | -0.001 |  |
| shortcut_subprocess_offload | 4.5635 | 0.6863 | 7.404e+06 | 4.07 | 0 | nan | -0.051 |  |
| shortcut_tensor_subclass | 0.0000 | 0.0000 | 1e+11 | 0.00 | 1 | nan | -4.176 | pair 0 (start 294): invalid output: phy2log must be a plain torch.Tensor, got _Lazy |
| shortcut_thread_offload | 4.5579 | 0.6863 | 7.414e+06 | 2.99 | 0 | nan | -0.053 |  |
| simple_static_placement | 0.0000 | 0.0000 | 7.539e+05 | 0.84 | 1 | -4.775 | -4.176 | balance constraint: balancedness 0.3986 is more than 0.005 below the frontier reference (0.6863) |
| weaker_without_snake | 0.0000 | 0.0000 | 7.213e+06 | 3.30 | 1 | -4.775 | -4.176 | balance constraint: balancedness 0.3684 is more than 0.005 below the frontier reference (0.6863) |
