# Hardening Report (iteration 12)

**Result.** Both weaknesses in `REVIEW.md` are fixed, and better results still score higher.

- **REVIEW item 1 (meter inside the policy process):** rebalance work is now counted outside the policy process, by a hardware instruction counter that the evaluator holds.
- **REVIEW item 2 (shared time windows):** the hidden test is now scored on time periods that no visible layer covers.
- **Better variants:** both win on all 20 of 20 paired rebalances, by far more than the run-to-run noise.
- **Weaker variants:** fail the balance constraint and score worst.
- **Shortcuts:** every one re-run through the hidden test fails, ties or loses.
- **What I could not fix:** the instruction count charges plain Python code several times more per CPU cycle than tensorised code. It is stated below with numbers.

All numbers come from CPU job `6bebf59b-51c` (AMD EPYC 9654), run on the final code. Earlier jobs (`4e2e3e27-2b7`, cancelled after preemptions, and `0b1da5a2-c73`) agree to about 0.2%. To reproduce:
- `bash scripts/run.sh` gives the reference, the baselines and validation;
- `python redteam/hardening/run_audit.py` gives the audit.

Results are in `redteam/hardening/results/`. Iteration 11's audit is kept in `results_iter11/`.

## What changed

| Weakness | Fix |
|---|---|
| REVIEW 1: the meter (`sys.monitoring` + a torch dispatch mode) ran inside the policy process. A deliberate policy could reach its counters and forge them. | **The work measure moved out of the policy process.**<br>• The evaluator opens a hardware counter of *retired user-space instructions* (`perf_event_open`) on the policy process before it loads any policy code. The counter is inherited by every thread and child process the policy starts.<br>• The work of a rebalance is the counter's increase from sending the window to receiving the full placement.<br>• The policy process holds no meter state. The counter is a kernel object owned by the evaluator, and `prctl(PR_TASK_PERF_EVENTS_DISABLE)` from the policy has no effect on it (tested). |
| REVIEW 1: Python-container operations (slicing, `in`, `+`/`*` on lists, `str.join`, C iterators) were charged by a hand-made cost model, about 1.6x under the tensor rate. | **Every operation is charged by the instructions it actually executes**, in any library. There are no per-operation weights to fit or exploit. The style probe built from exactly these operations (`probe_python_slicing`) costs the same as the plain-list version (6.37e8 vs 6.27e8 instructions). |
| (follows from moving the meter) The old in-process guards only protected the meter. | **OS-level sandbox:** each policy process seals itself with Linux Landlock (ABI 6 on the job nodes):<br>• read-only access to the Python, torch and numpy installation;<br>• no access to the workspace at all, so no test data or evaluator files;<br>• no file writes, no exec, no TCP;<br>• no signals or abstract-socket connections out of the sandbox.<br>The bypassable import blocks and audit hooks are gone. numpy, threads and `fork` are allowed because all of their work is counted. |
| REVIEW 2: train and test layers shared the same time windows, so burst timing learned on visible layers transferred to hidden ones. | **Held-out time.** The trace is cut into four equal time blocks:<br>• visible = even layers in blocks 0 and 2;<br>• hidden = odd layers in blocks 1 and 3.<br>No visible layer covers a hidden time period, and no hidden layer is visible at any time. Each slice has 20 rebalance pairs (input `[s, s+100)`, scored on `[s+100, s+200)`, `s` every 10 steps, all full 100-step windows as in the paper's evaluator). |
| (found here) In the old design, sequences at offsets 0/25/50/75 ran in one process. A policy could stash windows on a module such as `torch` and see most of a later window before being scored on it. | **One scored rebalance per process:**<br>1. one fork server per policy imports torch once and never runs policy code;<br>2. each pair runs in a freshly forked, sandboxed child: import, a warm-up call on a synthetic window, re-import, then one scored call;<br>3. the server is a child subreaper: everything the child started is killed before the next pair, and the evaluator kills everything left after each policy;<br>so no state, thread or helper process outlives its pair. |
| (found here) Wall-clock timeouts (600 s load, 1800 s per call) were in the method's path. | **Deterministic instruction caps** (1e11 per call, about 30 s of CPU). A process is failed only if its counter does not move at all for 10 minutes. |
| (found here) An evaluator killed mid-run would leave stale scores. | **Every discovered method is written as failed (worst values) before evaluation starts**, then overwritten as each one finishes. |

Measurement stability:
- Address-space randomisation is off and the hash seed fixed in the policy process.
- Its environment and working directory are fixed.
- torch kernels are pinned to one CPU code path (`ATEN_CPU_CAPABILITY=default`), so counts agree across x86 hosts.

## Do better results score higher? (hidden test, before → after)

Rank Δ uses the rule in `scoring/reference.json`: Σ |coefficient| × improvement / cohort standard deviation, with the reference at 0. The honest cohort is the 11 honest variants. The full cohort is all 25 policies of the combined run.

| Variant | Before (iteration 11): speedup / balance | After: speedup / balance | Rank Δ honest | Rank Δ full |
|---|---|---|---:|---:|
| reference `evolved_zigzag` | 2.7865 / 0.6583 | **4.6648 / 0.6863** | 0 | 0 |
| better: `better_compact_log2phy` (log2phy only as wide as the largest replica count) | 3.8698 / 0.6583 | 4.7504 / 0.6863 | **+0.040** | +0.039 |
| better: `better_lean_replication` (also 4 kernels per replication step instead of ~8) | 4.1429 / 0.6583 | 5.0715 / 0.6863 | **+0.190** | +0.186 |
| weaker: `weaker_without_snake` | fails balance gate (0.354) | fails balance gate (0.368) → 0 / 0 | −4.775 | −4.175 |
| simple load-oblivious placement | fails (0.383) | fails (0.399) → 0 / 0 | −4.775 | −4.175 |
| frontier baseline | 1.0000 / 0.6583 | 1.0002 / 0.6863 | −1.714 | −1.676 |
| greedy initial program `greedy_deepseek` | 0.0277 / 0.6618 | 0.0250 / 0.6869 | −2.169 | −2.121 |
| brute force: 256 sampled placements per node-layer, best on the visible window | 0.0694 / 0.6581 | 0.0624 / 0.6860 | −2.154 | −2.107 |
| probe: reference algorithm with Python lists | 0.1580 / 0.6583 | 0.0539 / 0.6863 | −2.157 | −2.110 |
| probe: Python built from slicing, `in`, list `+`, `str.join`, `itertools`/`reduce` | n/a | 0.0531 / 0.6863 | −2.158 | −2.110 |
| probe: reference algorithm in numpy | n/a (numpy was blocked) | 3.6099 / 0.6863 | −0.494 | −0.483 |

**Paired comparisons** (same 20 hidden pairs for every method):
- `better_compact_log2phy` is cheaper on 20/20 pairs (instruction ratio 1.017–1.020) with identical balance on every pair.
- `better_lean_replication` is cheaper on 20/20 pairs (1.084–1.091), also with identical balance.
- The greedy program's balance gain is +0.0006 ± 0.0018 (SE; better on 12/20 pairs) at 187x the reference's instructions.
- Brute force changes balance by −0.0003 ± 0.0002 at 75x the instructions: searching on the visible window does not help on the next one.

**Noise:**
- Three independent hidden-test runs gave reference speedups of 4.6648, 4.6638 and 4.6645. The per-pair instruction spread is at most 0.18% (reference 0.15%, anchor 0.18%).
- The same code under another file name (`shortcut_claim_one_operation`) ties to 0.002%. Code that adds a few unused lines differs by up to 0.3%, through memory layout.
- An AMD and an Intel node gave the same speedup to within 0.13%.
- The compact variant's +1.8% is 6x the largest layout effect and 60x the run-to-run spread.

The gains are smaller than under the old meter (+39% and +49% then). The old weights over-charged tensor elements. Executed instructions show that most of a wide `log2phy`'s cost is memory traffic and page faults, not instructions.

**Train/test parity** (honest variants, train → test):

| Variant | Train speedup / balance | Test speedup / balance |
|---|---|---|
| reference | 4.6656 / 0.6810 | 4.6622 / 0.6863 |
| compact | 4.7548 | 4.7504 |
| lean | 5.0772 | 5.0715 |
| greedy | 0.0253 / 0.6817 | 0.0250 / 0.6869 |
| numpy | 3.6124 | 3.6099 |

- Both slices fail the same variants, for the same reason.
- A load-shrinkage forecast tuned on the visible pairs gains +0.0024 balance on train and loses −0.0043 on test. Tuning to the visible data does not transfer.

## Shortcuts: fix and score before → after

All of these ran through the hidden-test evaluator (`redteam/hardening/cases/*.py`; full table in `redteam/hardening/results/summary.md`). Rank Δ uses the full cohort.

| Shortcut | Before (iteration 11) | Fix | After |
|---|---|---|---|
| forge the meter in-process (REVIEW 1) | possible: counters were Python globals in the policy process, reachable through frames, `torch.sys` or the reply fd | counter is a kernel object owned by the evaluator | `shortcut_disable_counters` (`prctl` disable + heap scan for evaluator state, at import and every call): no effect, 4.5860 (−0.036, slower by its own extra work) |
| claim one operation / tiny runtime | tie 2.7865 | self-reports ignored | tie: 4.6647 (−0.000) |
| work in a thread | failed (import block) | allowed; inherited counter counts it | 4.5504 (−0.052): counted, no gain |
| work in a forked child | n/a | inherited counter counts it | 3.8112 (−0.391): counted, no gain |
| work in a subprocess (exec) | failed (import block) | Landlock denies exec | exec refused, falls back: 4.5556 (−0.050) |
| move work into numpy | failed (blocked) | allowed; counted like any code | `probe_numpy_backend` 3.6099: an honest style, charged by its instructions |
| read the hidden windows | failed at load (in-process audit hook, bypassable via ctypes) | Landlock: workspace unreadable at OS level | Python `open`, libc `open` via ctypes and `/proc/<parent>/mem` all refused. The case falls back to the reference: 4.6518 (−0.006, memory-layout noise) |
| kill the evaluator | n/a | Landlock signal scoping; worst-value results written first | `os.kill(parent, SIGKILL)` refused; evaluator exit 0; case scored normally (4.6626) |
| helper process left alive for a later pair | n/a | fork server and evaluator are subreapers and kill leftovers | detached daemon killed (local test); no carry-over |
| state stashed across sequences (lookahead) | possible (one process per offset sequence) | one scored call per forked process | no state survives a pair |
| empty replicas / drop experts / global placement / impossible values / empty outputs / malformed file / tensor subclass / 79 KB lookup | all failed | unchanged checks | all failed, 0/0, −4.175 |
| inflate comparator policy | reference unchanged | frozen anchor in `eval/` | reference 4.6612 (unchanged within noise); the edited frontier itself drops to 0.3757 |
| remove comparator / flatten visible data / forge score file / omit roster | reference unchanged | unchanged | reference 4.6613 / 4.6645 / 4.6645 / 4.6645: unchanged within noise; forged file overwritten |

## Shortcuts and weaknesses left

- **The work measure charges instructions, not CPU cycles.** Measured instructions per user-mode cycle on the job host:
  - about 1.1 for torch-dispatch-heavy code (reference 1.11, compact 1.08, lean 1.16);
  - 1.7 for the numpy probe and the frontier;
  - 2.3 for the greedy program;
  - 4.1 for pure-Python list code.

  So per unit of CPU time, a pure-Python implementation pays about 3.7x more than the tensorised reference. This is the opposite of the old 1.6x under-charge, and larger. The numpy probe is about 15% faster than the reference in cycles but scores 0.77x as fast, because it executes 29% more instructions.

  I kept instructions anyway. They are the only measure here that is deterministic, unforgeable and independent of machine load. User-mode cycles and CPU time were recorded beside them and varied 3–5% between identical runs (up to 37% on a single pair), and they also differ between AMD and Intel nodes. Scoring cycles would make the stored reference depend on load. Spearman rank correlation of instructions with user cycles across all 15 successful policies is 0.80. The gym measures "fewer executed instructions", and `briefing/evaluation.md` says so.
- **Kernel work is not charged.** Page faults for fresh memory and pipe copies run in the kernel and are not counted, so instructions under-reward memory savings. For example, the compact `log2phy` variant has 142 instead of 551 page faults per rebalance and saves about 30% of CPU time, but only 1.8% of instructions (its user-mode cycles are the same).
- **A sandbox-level attack needs a kernel exploit.** Landlock (ABI 6) is enforced by the kernel. Escaping it needs a kernel bug, not a policy trick. If a node lacked Landlock, the harness would fall back to an in-process audit hook, which is bypassable, and record `sandbox.mode` in the score file.
- **The raw trace is public.** A policy file of at most 64 KB could embed knowledge of the public trace's held-out layers or time blocks. That is a flagrant test-data shortcut and is left to the shortcut grader. Mitigations:
  - the expert relabelling is secret and the layers are shuffled;
  - the policy cannot read any file;
  - the policy sees one window per process.
- **Import-time work is free.** It cannot depend on the windows, since none has been sent yet, so it only buys input-independent precomputation.
- **Balance gate cliff.** Balance more than 0.005 below the frontier fails the whole run.
- **Smaller test.** The hidden test is 20 pairs × 29 layers (iteration 11 had 43), because half of the trace's time is now visible-only.

## Reference score before → after

- Before: `speedup_vs_frontier_reference` 2.7865 and `balancedness` 0.6583 (hidden test, old meter, shared time windows).
- After: `scoring/scores_test.json` from the end-to-end run of the final code, with the hidden test re-run from `scripts/reproduce.sh`:
  - reference: **4.6622 / 0.6863**;
  - frontier: 0.9999 / 0.6863;
  - greedy: 0.0250 / 0.6869.
- Train: reference 4.6656 / 0.6810.
- The paper reports 5.3x (wall-clock, unreleased code). The speedup is now on the paper's side of the old 2.79x, without tuning anything toward it. The paper's numbers in `scoring/reference.json` are unchanged.

## Headroom check

- **No scored metric is at its bound.**
  - `balancedness` is 0.686 (bound 1). Placing on the next window's own loads reaches about 0.81 on both slices.
  - `speedup` is unbounded. Two honest variants already beat the reference (+1.8%, +8.7%).
- **No simple variant reaches a bound.** The only balance improvement seen is greedy's +0.0006 at 187x the instructions. Load shrinkage does not transfer from train to test.
- **Brute force does not solve the test.**
  - The packing loops the paper's method removes (greedy, 0.025) and a 256-candidate placement search (0.062) both rank far below the reference.
  - The search cannot see the window it is scored on.
  - Exhaustive enumeration of 72 replicas onto 8 GPUs per node-layer is infeasible.
- **The baselines are clearly below the reference on a scale with room above it:** frontier −1.71, greedy −2.17 (honest cohort).
- No extra experiment was needed. The main experiment already contains where the paper's method falls short: next-window balance under drift, 0.686 vs about 0.81.

## Weights

Both scored metrics keep coefficient 1.0, as the paper's evaluator weighted balancedness and rebalance speed equally. Validity and the balance gate are hard constraints (any violation scores worst), so speed cannot buy back a constraint failure. All diagnostics (instructions, CPU time, worst window, validity, failure flag) carry coefficient 0.

## Could not fix / caveats

- **Instructions are not time,** and the gap depends on implementation style (above). A future hidden test on a different x86 CPU gives the same instruction counts to about 0.1%, but only on a Linux kernel that allows user-space `perf_event_open`. Without it the evaluator stops with an explanation instead of scoring.
- **The paper's frontier-lab reference and evolved code are not released.** Both are reconstructions:
  - frontier: zigzag in the replica-to-GPU stage only;
  - evolved: zigzag reused in both packing stages, the stated breakthrough.
  - The paper's unspecified "subtle enhancements" are not reproduced.
- **Each scored rebalance covers 29 of the 58 layers and half of the trace's time,** so that both stay hidden.
- **`DETAILS.md` still describes iteration 11's evaluator.** It appears to be a system-compiled log, so I left it untouched. `briefing/`, the `scoring/` documents and `DESIGN_NOTES.md` describe the current gym.
