#!/bin/bash
set -e

cd /home/user

python data/generate_eplb_traces.py
python baseline/eplb_baseline.py --traces data/eplb_traces.json --output outputs/eplb/greedy_deepseek.json
python method/eplb_evolved.py --traces data/eplb_traces.json --output outputs/eplb/evolved_zigzag.json
bash scripts/evaluate.sh
