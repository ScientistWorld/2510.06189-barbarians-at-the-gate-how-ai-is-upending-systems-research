#!/bin/bash
set -e

cd /home/user

mkdir -p outputs/eplb
rm -f outputs/eplb/*.json
python data/generate_eplb_traces.py
python baseline/eplb_baseline.py --traces data/eplb_traces.json --output outputs/eplb/greedy_deepseek.json
python baseline/eplb_frontier_reference.py --traces data/eplb_traces.json --output outputs/eplb/frontier_reference_zigzag.json
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
