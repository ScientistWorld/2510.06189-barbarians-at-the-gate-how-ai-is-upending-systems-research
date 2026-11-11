Train/Test Evaluation Design Notes (current, iteration 11)

- Data: the paper's expert-load trace (abmfy/eplb-openevolve). Visible = even MoE layers (`data/eplb_train_windows.json`); hidden = odd layers, secretly relabelled, only in `eval/test/test_windows.json` (rebuilt by `eval/test/prepare_test_windows.py`). Four window offsets, 43 rebalances per method.
- Interface: every `outputs/eplb/<name>.py` defines `rebalance_experts()`; `eval/{train,test}/evaluate.py` runs each in `harness.py` (sandboxed subprocess), streaming one window at a time and scoring on the next.
- Scored: `speedup_vs_frontier_reference` (work of the frozen anchor in `eval/*/anchor_frontier_reference.py` / method work) and `balancedness` (next-window mean/max GPU load). Hard constraints: valid hierarchical placement, determinism, guards, balance within 0.005 of the anchor; any failure scores 0/0.
- Work meter: bytecodes 1, C calls 1 (+2 per scanned Python element), torch kernels 20 (+0.06 per tensor element); calibrated to CPU time (Spearman 0.976). See `HARDENING.md`.
- eval/train and eval/test share no code (duplicated harness/anchor), and neither imports method/.
