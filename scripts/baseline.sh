#!/bin/bash
# Publish the baseline policies as submissions in outputs/eplb/.
set -e
cd /home/user
mkdir -p outputs/eplb
cp baseline/eplb_greedy_deepseek.py outputs/eplb/greedy_deepseek.py
cp baseline/eplb_frontier_reference.py outputs/eplb/frontier_reference_zigzag.py
