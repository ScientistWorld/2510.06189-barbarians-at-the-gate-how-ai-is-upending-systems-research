# Evaluation

## Metrics

- `mean_imbalance_factor`: average over traces of `(mean GPU load) / (max GPU load)`. Higher is better; 1.0 is perfectly balanced.
- `work_efficiency_vs_greedy`: greedy baseline mean operation count divided by candidate mean operation count. Higher is better. This replaces wall-clock runtime with a machine-independent work proxy.
- `mean_operation_count`: counted algorithmic operations for group packing, replication, and placement. Lower is better.
- `valid_fraction`: fraction of traces whose placement has the requested number of replicas, valid expert ids, and positive replica count for every logical expert. Higher is better.

## Baselines and Targets

The main baseline is the public DeepSeek EPLB-style greedy bin-packing policy. The paper reports an initial greedy runtime of about 540 ms with imbalance factor 0.66, a non-public reference runtime of 19.6 ms with the same imbalance, and an evolved runtime of 3.7 ms with the same imbalance. This environment reports the same comparison as relative work efficiency rather than raw runtime.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to generate deterministic expert-load traces, run the baseline and candidate method, and write `scoring/scores.json`. Standalone evaluation is `bash scripts/evaluate.sh`; it reads placement artifacts from `outputs/eplb/` and does not import from `method/`.
