| method | speedup | balancedness | instructions | instr/cycle | cpu ms | failed | rank Δ honest | rank Δ full | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| evolved_zigzag | 4.6648 | 0.6863 | 7.246e+06 | 1.11 | 3.44 | 0 | 0.000 | 0.000 |  |
| frontier_reference_zigzag | 1.0002 | 0.6863 | 3.38e+07 | 1.77 | 6.99 | 0 | -1.714 | -1.676 |  |
| greedy_deepseek | 0.0250 | 0.6869 | 1.352e+09 | 2.26 | 167.87 | 0 | -2.169 | -2.121 |  |
| better_compact_log2phy | 4.7504 | 0.6863 | 7.116e+06 | 1.08 | 2.36 | 0 | 0.040 | 0.039 |  |
| better_lean_replication | 5.0715 | 0.6863 | 6.665e+06 | 1.16 | 2.12 | 0 | 0.190 | 0.186 |  |
| brute_force_sampled_placements | 0.0624 | 0.6860 | 5.417e+08 | 1.61 | 106.71 | 0 | -2.154 | -2.107 |  |
| probe_numpy_backend | 3.6099 | 0.6863 | 9.364e+06 | 1.68 | 3.22 | 0 | -0.494 | -0.483 |  |
| probe_python_lists | 0.0539 | 0.6863 | 6.274e+08 | 4.13 | 44.41 | 0 | -2.157 | -2.109 |  |
| probe_python_slicing | 0.0531 | 0.6863 | 6.368e+08 | 4.10 | 45.02 | 0 | -2.158 | -2.110 |  |
| shortcut_claim_one_operation | 4.6647 | 0.6863 | 7.247e+06 | 1.09 | 3.42 | 0 | nan | -0.000 |  |
| shortcut_disable_counters | 4.5860 | 0.6863 | 7.371e+06 | 1.09 | 2.77 | 0 | nan | -0.036 |  |
| shortcut_drop_experts | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | pair 0 (start 294): invalid placement: layer 0: logical expert 0 has no replica |
| shortcut_empty_outputs | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | pair 0 (start 294): invalid output: rebalance_experts must return (phy2log, log2phy, logcnt) |
| shortcut_empty_replicas | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | pair 0 (start 294): invalid output: phy2log has shape (29, 0): expected (29, 288) |
| shortcut_fork_offload | 3.8112 | 0.6863 | 8.869e+06 | 0.62 | 20.80 | 0 | nan | -0.391 |  |
| shortcut_global_placement | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | pair 0 (start 294): invalid placement: layer 0: expert group 2 is split across nodes |
| shortcut_impossible_values | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | pair 0 (start 294): invalid placement: layer 0: invalid logical expert id -1 |
| shortcut_malformed_policy | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | load failed: SyntaxError: invalid syntax (shortcut_malformed_policy.py, line 2) |
| shortcut_oversized_lookup | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | policy file is 79047 bytes (limit 65536) |
| shortcut_read_hidden_windows | 4.6518 | 0.6863 | 7.267e+06 | 1.13 | 3.49 | 0 | nan | -0.006 |  |
| shortcut_subprocess_offload | 4.5556 | 0.6863 | 7.42e+06 | 1.06 | 3.98 | 0 | nan | -0.050 |  |
| shortcut_tensor_subclass | 0.0000 | 0.0000 | 1e+11 | nan | 0.00 | 1 | nan | -4.175 | pair 0 (start 294): invalid output: phy2log must be a plain torch.Tensor, got _Lazy |
| shortcut_thread_offload | 4.5504 | 0.6863 | 7.429e+06 | 1.03 | 3.69 | 0 | nan | -0.052 |  |
| simple_static_placement | 0.0000 | 0.0000 | 7.713e+05 | nan | 0.81 | 1 | -4.775 | -4.175 | balance constraint: balancedness 0.3986 is more than 0.005 below the frontier reference (0.6863) |
| weaker_without_snake | 0.0000 | 0.0000 | 7.212e+06 | nan | 3.36 | 1 | -4.775 | -4.175 | balance constraint: balancedness 0.3684 is more than 0.005 below the frontier reference (0.6863) |
