# Evaluation

## Metrics

- `mean_imbalance_factor`: average over traces of `(mean GPU load) / (max GPU load)`. Higher is better; 1.0 is perfectly balanced.
- `work_efficiency_vs_greedy`: lightweight reference mean operation count divided by candidate mean operation count. Higher is better. This replaces wall-clock runtime with a machine-independent work proxy.
- `work_efficiency_vs_frontier_reference`: stronger benchmark-reference mean operation count divided by candidate mean operation count, multiplied by the candidate's measured balance. Higher is better and is the primary metric because the target problem is to reduce rebalance work without losing placement quality.
- `mean_operation_count`: counted algorithmic operations for group packing, replication, and placement. Lower is better.
- `valid_fraction`: fraction of required traces whose placement has the requested number of replicas, valid expert ids, and positive replica count for every logical expert. Higher is better.
- `complete_fraction`: fraction of required traces for which a method emits an artifact. Higher is better.

## Baselines and Targets

Reference artifacts provide two fixed comparison anchors: one lightweight anchor for basic sanity checking and one stronger anchor for the primary score. Published wall-clock values are kept only as unscored context in `reported_runtime_ms`; the scored work metrics use operation-count ratios from the reproducible benchmark because wall-clock runtime is hardware-dependent.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to generate deterministic expert-load traces, produce placement artifacts, and write `scoring/scores.json`. Standalone evaluation is `bash scripts/evaluate.sh`; it reads placement artifacts from `outputs/eplb/` and does not import from any candidate method implementation.
