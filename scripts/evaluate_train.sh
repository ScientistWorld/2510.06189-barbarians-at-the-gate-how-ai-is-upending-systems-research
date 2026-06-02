#!/bin/bash
set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py --traces data/eplb_traces.json --outputs-dir outputs/eplb --scores scoring/scores_train.json
