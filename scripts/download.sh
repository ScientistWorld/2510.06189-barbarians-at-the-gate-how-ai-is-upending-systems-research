#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This reproduction has no external data or model downloads.

set -e

cd /home/user

# The paper's ShareGPT/GSM8K expert-load traces are not released. This
# reproduction uses a deterministic tracked generator and needs no downloads.
python data/generate_eplb_traces.py
