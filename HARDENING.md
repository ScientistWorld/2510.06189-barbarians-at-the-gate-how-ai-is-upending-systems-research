# Hardening Report (iteration 11)

**Result.** Better results now score higher, and every major shortcut in REDTEAM.md is closed. The two shortcuts that do not fail (self-reporting work, deleting or slowing the comparator) now tie or leave the reference's score unchanged instead of paying.

The gym was rebuilt on the paper's own data and evaluator protocol:
- the expert-load trace released with the paper's EPLB case study,
- 288 replicas, 8 groups, 4 nodes, 32 GPUs,
- rebalance every 100 steps, scored on the next window.

On this setup the reproduction shows the paper's claim in the paper's direction:
- **Reference:** the evolved zigzag policy needs **2.79x** less rebalance work than the frontier reference (3.3x in CPU time; the paper reports 5.3x in §5.2 and 2x in Table 1).
- **Balance:** it keeps **identical** next-window GPU balance (0.6583 test / 0.6674 train), on the paper's scale (0.66).
- **Greedy initial program:** about 36x more work than the frontier and about 100x more than the reference, with slightly better balance (0.6618).

All numbers below come from the two CPU jobs (`ec4cef73-4b8`, preempted after most of the audit; `5ee210d1-272`, complete) and are reproducible with `python redteam/hardening/run_audit.py`.

## What was wrong and what changed

| Problem (REDTEAM.md) | Fix |
|---|---|
| Work was a number the method wrote (`operation_count`); "claim one operation" won (+1.22 rank). | A method is now a policy file implementing the paper's `rebalance_experts()`. The hidden test **runs** it in a sandboxed process and measures the work of each call itself. Self-reported values are ignored. |
| The comparator was an editable artifact; inflating it won (+4.03), deleting it dropped the primary metric. | Speedup is measured against a **frozen copy of the frontier policy inside `eval/test/`** (and `eval/train/`), re-run on the same windows in every evaluation. |
| Empty placements and dropped experts scored perfect balance (+0.097). | Exact validity checks: 288 slots, every expert ≥ 1 replica, `logcnt` matches `phy2log`, `log2phy` consistent and `-1`-padded. Any invalid window fails the run. |
| Node/group assignment was ignored. | **Node locality enforced:** each expert group lives on one node, and every node hosts exactly 2 groups (the paper's three-stage hierarchical EPLB). Global placement now fails. It would otherwise raise balance from 0.666 to about 0.755. |
| Test traces were a fully visible fixed pool (cache, flatten). | Hidden test = the 29 **held-out MoE layers**, secretly relabelled (groups permuted, experts permuted within groups, layers shuffled), present only in `eval/test/`. The policy process receives one window at a time and never holds the window it is scored on. |
| Synthetic traces were too easy (balance 0.99, greedy near-perfect, a baseline outranked the reference). | The paper's real trace and protocol: greedy and frontier land at the paper's 0.66. The next-window oracle reaches about 0.81, so balance has headroom. |
| Missing values were skipped, empty outputs got best-value spread, malformed files left stale scores, and the roster came from editable train scores. | Any failure writes **worst values (0, 0)** in both scored metrics and records the reason under `_failures`. The evaluator never aborts on a bad policy. Every `outputs/eplb/*.py` on disk is scored, with no roster. |
| Wall-clock runtime replaced by a hand-specified op proxy that charged sorts/tensors inconsistently. | Deterministic work meter over the actual execution (details below), calibrated against measured CPU time. |

Further guards: no threads or processes, no file I/O, only `torch` plus the computational stdlib may be imported (no numpy, ctypes, sys, os, gc, importlib, ...), a 64 KB file limit, no TorchScript or `torch.compile`, no `sys.monitoring` changes, no torch dispatch-mode bypasses, exact `torch.Tensor` outputs, no dispatch mode left active, and a metered run must match an unmetered replay exactly (determinism).

**Work meter.** It counts in the policy process:
- Python bytecodes (any thread): 1 unit each,
- calls into C functions: 1 unit each, plus 2 units per element of the Python container they scan (sorts `n log2 n`),
- PyTorch kernels: 20 units each, plus 0.06 units per tensor element read or written (views free, sorts × `log2 n`).

The first job used a-priori weights (kernel 400, tensor element 0.1). Its calibration check showed those weights scored a pure-Python version of the reference as *faster* than the frontier although it is 9x slower in CPU time, a "worse method scores higher" defect. The weights were refit once against unmetered CPU time of all honest policies, and the saved counters were rescored. Job 2 confirmed the result:
- Spearman(work, CPU) = **0.976**, and units per CPU-ms lie in 43k–77k for every honest policy.
- Raw counters were **bit-identical** across the two jobs for all 11 successful policies, so seed and run noise is zero.

## Do better results score higher? (hidden test, before → after)

The rank is the sum over scored metrics of |coefficient| × improvement / cohort standard deviation, with the reference at 0. The scored metrics are `speedup_vs_frontier_reference` (primary) and `balancedness`, coefficient 1.0 each.

| Variant | Before (old gym) | After: speedup / balancedness | Rank Δ, honest cohort | Rank Δ, full cohort |
|---|---|---|---:|---:|
| reference `evolved_zigzag` | eff 6.629 / bal 0.9878 | **2.7865 / 0.6583** | 0 | 0 |
| better: `better_compact_log2phy` (log2phy only as wide as max replica count) | (old better variant: +0.0075) | 3.8698 / 0.6583 | **+0.653** | +0.871 |
| better: `better_lean_replication` (also 4 kernels per replication step instead of ~8) | n/a | 4.1429 / 0.6583 | **+0.817** | +1.090 |
| weaker: `weaker_without_snake` | 4.065 / 0.594 (−4.74) | fails balance gate (0.354 measured) → 0 / 0 | −4.083 | −4.406 |
| frontier baseline | eff 1.0 (anchor) | 1.0000 / 0.6583 | −1.076 | −1.436 |
| greedy initial program `greedy_deepseek` | **outranked the reference** in the full cohort (+0.076) | 0.0277 / 0.6618 | −1.650 | −2.207 |
| simple load-oblivious placement | n/a | fails balance gate (0.383) → 0 / 0 | −4.083 | −4.406 |
| brute force: 256 sampled placements per node-layer, best on the visible window | 0.0049 (−3.23) | 0.0694 / 0.6581 | −1.638 | −2.185 |
| probe: reference algorithm written with Python lists | n/a | 0.1580 / 0.6583 | −1.584 | −2.113 |

**Paired comparisons** (same 43 hidden rebalances for every method):
- `better_compact_log2phy` is cheaper on 43/43 windows (work ratio 1.33–1.42), with balance difference exactly 0 on every window.
- `better_lean_replication` is cheaper on 43/43 windows (ratio 1.43–1.52), also with balance difference exactly 0.
- Because the test is deterministic, the improvement exceeds its noise (0) by construction, and the gaps are far larger than the 1.8x residual spread of the work measure.
- Greedy's balance gain is +0.0034 ± 0.0012 (SE; better on 34/43 windows) at 100x the work.
- The brute-force search changes balance by −0.0002 ± 0.0003 and wins 5/43 windows: searching on the visible window does not help on the next one.

**Train/test parity:** identical speedups for every honest variant on train (reference 2.7865, compact 3.8887, lean 4.1645, greedy 0.0277). Balancedness is comparable (0.6674 train vs 0.6583 test). Both slices fail the same variants for the same reason.

## Shortcuts: fix and score before → after

All of these ran through the hidden-test evaluator (`redteam/hardening/cases/*.py`; full table in `redteam/hardening/results/summary.md`). Rank Δ uses the full cohort.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| claim one operation / tiny runtime | 2015 eff, +1.222 | work measured, not reported | **tie**: 2.7865 / 0.6583, rank 0 (attributes ignored) |
| empty replicas | +0.097 | exact budget + coverage | failed, 0/0, −4.41 |
| drop experts (hottest expert everywhere) | +0.097 | coverage | failed, −4.41 |
| inflate comparator artifact | +4.027 | frozen anchor in `eval/` | reference unchanged (2.7865); the edited frontier itself drops to 0.367 |
| remove comparator | primary metric missing (tie) | frozen anchor | reference unchanged, metric present |
| visible trace cache | tie | held-out relabelled layers, file reads blocked, 64 KB limit | `shortcut_read_hidden_windows` fails at load; `shortcut_oversized_lookup` rejected; −4.41 |
| flatten visible workload | +0.097 | test never reads `data/` | reference unchanged |
| ignore node groups | tie (unused) | node locality enforced | `shortcut_global_placement` fails ("expert group split across nodes"), −4.41 |
| omit method from roster | missing, tie | every policy on disk scored | reference scored without `scores_train.json` |
| forged summary / flags | already failed | score file always rewritten | overwritten, reference unchanged |
| empty outputs | not worst in every metric | worst in every scored metric | failed, 0/0, −4.41 |
| impossible values | failed | invalid maps fail the run | failed, −4.41 |
| malformed artifact | exit 1, stale scores | load failure scored worst, exit 0 | failed, 0/0, −4.41 |
| *new*: work in a thread | n/a | import block + thread guard | failed at load |
| *new*: work in a subprocess | n/a | import block + audit hook | failed |
| *new*: work in numpy | n/a | `Tensor.numpy` blocked, numpy import blocked | failed |
| *new*: TorchScript | n/a | `torch.jit.*` / `torch.compile` blocked | failed at load |
| *new*: switch off the bytecode meter | n/a | `sys.monitoring` setters guarded | failed |
| *new*: pop the kernel meter's dispatch mode | n/a | mode/dispatch-key bypass APIs guarded from policy code | failed |
| *new*: leave a dispatch mode active (to run code outside the metered call) | n/a | dispatch-stack check after each call | failed (the harness process aborts before the check; still worst values) |
| *new*: tensor subclass with hookable conversions | n/a | exact-type output check | failed (torch refuses to create the subclass under the meter; the type check is a second line) |

## Shortcuts and weaknesses left

- **Deliberate in-process tampering.** The policy runs in the same process as the meter. Every simple route is blocked or detected, but a policy that deliberately reaches evaluator internals could still forge counters. Examples: through `torch.sys`/`torch.__builtins__`, through traceback frames, or by writing to the reply file descriptor. Closing this needs an OS-level sandbox. It is flagrant and left to the shortcut grader.
- **Work-measure blind spots.** Single bytecodes that do O(n) work (list slicing, `in`, `+`/`*` on sequences), C iterators consumed by C code (`itertools`, `functools.reduce`) and `str.join` are under-charged. Python-container code is still about 1.6x under-charged relative to tensor code (48k vs 77k units per CPU-ms). A policy built to exploit this would show work units far below its `diagnostic_cpu_ms_per_rebalance`, which is recorded unscored for the grader.
- **The raw trace is public.** An agent with internet could fetch the held-out layers. Mitigations: secret relabelling and layer shuffle, blocked file reads, and the 64 KB limit. A compact embedded fingerprint table would remain possible. Using it is a flagrant test-data shortcut.
- **Shared timeline.** Train and test layers share the same time windows (bursts at the same steps). A stateful policy could learn when bursts happen from visible layers. Four window offsets reduce but do not remove this.
- **Import-time work is free.** It cannot depend on the windows, so it only buys input-independent precomputation.
- **Balance gate cliff.** Balance more than 0.005 below the frontier fails the whole run.
- **Cohort normalization.** Within the gate, balance gains count through cohort-normalized ranks, where failed runs (balance 0) dominate the standard deviation. In the audited cohort, a 1% relative balance gain and a 1% relative speed gain are worth about the same (0.022 vs 0.023 rank per %).

## Reference score before → after

- Before: `work_efficiency_vs_frontier_reference` 6.629, `mean_imbalance_factor` 0.9878, self-reported work 304, on synthetic traces.
- After (hidden test): `speedup_vs_frontier_reference` **2.7865**, `balancedness` **0.6583**. Train gives 2.7865 / 0.6674.
- These are the only methods in `scoring/scores_test.json` / `scores_train.json`, together with the baselines: frontier 1.0 / 0.6583, greedy 0.0277 / 0.6618.

## Headroom check

- **No scored metric is at its bound.** `balancedness` 0.658 (bound 1). An oracle that places on the next window's own loads reaches about 0.81 (0.80–0.83 depending on packing and layer half; exploratory probe on the raw trace, offset 0). `speedup` is unbounded, and two honest variants already exceed the reference by 39–49%.
- **No simple variant reaches the bound.** Simple forecasts tried on visible layers did not exceed the reference's 0.6668 (best 0.6665; most lower, down to 0.625): load shrinkage, power transforms, EMA over windows, two-window sums, mixed normalised history. Balance headroom needs real forecasting research.
- **Brute force does not solve the test.** The packing loops the paper's method removes (greedy, 0.028) and a 256-candidate placement search (0.069) both rank far below the reference. Exhaustive enumeration of 72 replicas onto 8 GPUs per node-layer is infeasible. The search cannot see the window it is scored on.
- The old gym's `valid_fraction`/`complete_fraction` (both at bound) are now unscored diagnostics. Validity is enforced as a hard constraint instead.
- No extra experiment was needed. The main experiment already includes where the paper's method falls short (next-window balance under drift: 0.66 vs about 0.81).

## Weights

Both scored metrics have coefficient 1.0. The paper's evaluator used an equally weighted combination of balancedness and rebalance speed. Validity and balance-no-worse are hard constraints (any violation scores worst) rather than weighted terms, so speed cannot buy back a constraint failure. All diagnostics (work units, CPU time, worst window, validity, failure flag) carry coefficient 0.

## Could not fix / caveats

- The paper's frontier-lab reference and its evolved code are not released. Both are reconstructions:
  - Frontier: zigzag in the replica-to-GPU stage only, consistent with the reported 540 → 19.6 ms.
  - Evolved: zigzag reused in both packing stages, the paper's stated breakthrough.
  - The paper's unspecified "subtle enhancements" are not reproduced. Its 5.3x becomes 2.79x here (3.3x CPU).
- Wall-clock runtime is replaced by deterministic work units. Their calibration was fit on one CPU type (job host) and tracks CPU time within about 1.8x across very different implementation styles, not exactly.
- Each scored rebalance covers 29 of the 58 layers, so that half the layers can stay hidden.
- Older artifacts under `redteam/` (the pre-hardening audit, including `audit.py`) target the previous artifact interface and are kept as history only.
