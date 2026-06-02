#!/bin/bash
set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --traces data/eplb_traces.json \
  --outputs-dir outputs/eplb \
  --scores scoring/scores_test.json \
  --train-scores scoring/scores_train.json
