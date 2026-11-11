#!/bin/bash
# Publish the reproduced method as a submission in outputs/eplb/.
set -e
cd /home/user
mkdir -p outputs/eplb
cp method/eplb_evolved.py outputs/eplb/evolved_zigzag.py
