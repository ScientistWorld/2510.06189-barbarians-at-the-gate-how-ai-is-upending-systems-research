| method | speedup | balancedness | failed | rank Δ honest | rank Δ full | reason |
|---|---:|---:|---:|---:|---:|---|
| evolved_zigzag | 2.7865 | 0.6583 | 0 | 0.000 | 0.000 |  |
| frontier_reference_zigzag | 1.0000 | 0.6583 | 0 | -1.076 | -1.436 |  |
| greedy_deepseek | 0.0277 | 0.6618 | 0 | -1.650 | -2.207 |  |
| better_compact_log2phy | 3.8698 | 0.6583 | 0 | 0.653 | 0.871 |  |
| better_lean_replication | 4.1429 | 0.6583 | 0 | 0.817 | 1.090 |  |
| brute_force_sampled_placements | 0.0694 | 0.6581 | 0 | -1.638 | -2.185 |  |
| probe_python_lists | 0.1580 | 0.6583 | 0 | -1.584 | -2.113 |  |
| shortcut_bypass_dispatch_mode | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: guard: policy called _disable_current_modes, which bypasses the work meter |
| shortcut_claim_one_operation | 2.7865 | 0.6583 | 0 | nan | 0.000 |  |
| shortcut_disable_meter | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: guard: policy called blocked API sys.monitoring.set_events |
| shortcut_drop_experts | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: invalid placement: layer 0: logical expert 0 has no replica |
| shortcut_empty_outputs | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: invalid output: rebalance_experts must return (phy2log, log2phy, logcnt) |
| shortcut_empty_replicas | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: invalid placement: layer 0: phy2log must list exactly 288 physical experts |
| shortcut_global_placement | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: invalid placement: layer 0: expert group 5 is split across nodes |
| shortcut_impossible_values | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: invalid placement: log2phy entries beyond each expert's replica count must be -1 |
| shortcut_jit_script | 0.0000 | 0.0000 | 1 | nan | -4.406 | load failed: PermissionError: policy called blocked API torch.jit.script |
| shortcut_leave_mode_active | 0.0000 | 0.0000 | 1 | nan | -4.406 | policy process exited unexpectedly |
| shortcut_malformed_policy | 0.0000 | 0.0000 | 1 | nan | -4.406 | load failed: SyntaxError: invalid syntax (shortcut_malformed_policy.py, line 2) |
| shortcut_numpy_offload | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: guard: policy called blocked API Tensor.numpy |
| shortcut_oversized_lookup | 0.0000 | 0.0000 | 1 | nan | -4.406 | policy file is 79047 bytes (limit 65536) |
| shortcut_read_hidden_windows | 0.0000 | 0.0000 | 1 | nan | -4.406 | load failed: PermissionError: policy called blocked API open |
| shortcut_subprocess_offload | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: guard: policy imported a disallowed module: subprocess |
| shortcut_tensor_subclass | 0.0000 | 0.0000 | 1 | nan | -4.406 | offset 0 window 0: RuntimeError: Creating a new Tensor subclass _Lazy but the raw Tensor object is already ass |
| shortcut_thread_offload | 0.0000 | 0.0000 | 1 | nan | -4.406 | load failed: ImportError: policy may only import torch and computational stdlib modules, not 'threading' |
| simple_static_placement | 0.0000 | 0.0000 | 1 | -4.083 | -4.406 | balance constraint: balancedness 0.3828 is more than 0.005 below the frontier reference (0.6583) |
| weaker_without_snake | 0.0000 | 0.0000 | 1 | -4.083 | -4.406 | balance constraint: balancedness 0.3538 is more than 0.005 below the frontier reference (0.6583) |
