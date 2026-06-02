#!/bin/bash
set -e

cd /home/user

python data/generate_eplb_traces.py
python method/eplb_evolved.py --traces data/eplb_traces.json --output outputs/eplb/evolved_zigzag.json
bash scripts/evaluate.sh
