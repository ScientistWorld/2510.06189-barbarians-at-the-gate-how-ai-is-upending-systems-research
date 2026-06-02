Train/Test Evaluation Design Notes

Checked:
- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `scoring/reference.json`.
- Confirmed both slice evaluators read method artifacts from `outputs/eplb/*.json` instead of importing or rerunning `method/`.
- Confirmed `evaluate_test.sh` invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Confirmed `evaluate_train.sh` invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`.
- Confirmed there are no cross-imports between `eval/train/` and `eval/test/`.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Ran `bash -n` on both slice scripts.

Changed:
- Added family-level load-balance metrics to both train and test evaluators:
  `family_imbalance_floor` and `family_imbalance_spread`.
- Added high-skew trace metrics to both train and test evaluators:
  `high_skew_mean_imbalance_factor` and `high_skew_mean_operation_count`.
- Removed `reported_runtime_ms` from active metric definitions in `scoring/reference.json`; wall-clock values remain only in paper-result context because hardware-dependent runtime should not drive the gym score.
- Added required `higher_is_better`, `coefficient`, and `description` fields for every active metric in `scoring/reference.json`.
- Reweighted balance/validity coefficients slightly so the score better reflects the paper's claim: work reduction without sacrificing placement validity or load balance.

Problem fixed:
- The first-pass schema declared metrics that were not fully described and included a wall-clock metric as an active metric entry. This weakened static coverage checks and conflicted with the hardware-independent scoring requirement.
- The first-pass split mostly tested a global mean over an even/odd trace partition. The added family and high-skew metrics make visible-metric gaming harder by requiring improvements to hold across workload families and difficult load distributions.

Runtime concerns for the next iteration:
- This pass intentionally did not execute evaluators or produce numbers. The runner should verify that both slices emit all metrics listed in `scoring/reference.json`.
- The trace pool is compact: 42 traces total, 21 per slice, with 7 traces per family per slice. The new family/high-skew metrics help triangulate behavior, but the runner should watch for metric saturation on the existing reproduction.
- `eval/test/evaluate.py` uses `scoring/scores_train.json` when available to restrict test scoring to methods seen on train. If train scores are absent, it falls back to all artifacts in `outputs/eplb/`; the runner should check that this matches the final harness behavior.
