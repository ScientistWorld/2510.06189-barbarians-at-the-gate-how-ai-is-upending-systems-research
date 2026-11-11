#!/bin/bash
# Install Python packages into the container overlay.
#
# Run by the system on the login node (internet) inside the container.
# Install into the SYSTEM Python with uv; do NOT create a virtual environment.
#
# torch is pinned: the evaluator's work meter counts PyTorch kernel launches,
# so the torch version is part of the measurement.

set -e

uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" numpy
