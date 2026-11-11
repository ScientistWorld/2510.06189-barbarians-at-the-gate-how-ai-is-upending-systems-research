#!/bin/bash
# Score every policy in outputs/eplb/ on the visible (train) windows.
set -e
cd /home/user
mkdir -p scoring
python eval/train/evaluate.py --outputs-dir outputs/eplb --scores scoring/scores_train.json
