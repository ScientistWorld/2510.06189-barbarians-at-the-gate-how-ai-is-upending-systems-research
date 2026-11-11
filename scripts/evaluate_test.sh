#!/bin/bash
# Hidden test: score every policy in outputs/eplb/ on the held-out windows.
set -e
cd /home/user
mkdir -p scoring
python eval/test/evaluate.py --outputs-dir outputs/eplb --scores scoring/scores_test.json
