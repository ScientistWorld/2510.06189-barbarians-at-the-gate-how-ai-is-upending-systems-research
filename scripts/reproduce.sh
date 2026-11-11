#!/bin/bash
# End-to-end reproduction: publish baselines + method, score train and test.
set -e
cd /home/user
mkdir -p outputs/eplb
rm -f outputs/eplb/*.py outputs/eplb/*.json
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
