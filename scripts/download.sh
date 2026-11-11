#!/bin/bash
# Download all data needed for this environment.
#
# Run on a node with internet access BEFORE building the container or
# submitting compute jobs. Idempotent: skips work whose output exists.
#
# Estimated total download size: 0.23 GB (raw expert-load trace, deleted after extraction)
# Estimated disk usage after extraction: ~1.5 MB (data/eplb_train_windows.json, tracked in git)
#
# Source: the expert-load trace released with the paper's EPLB case study,
# https://huggingface.co/datasets/abmfy/eplb-openevolve (pinned revision).
# Only the visible (even) MoE layers are extracted into data/. The raw file
# also contains the held-out layers used by the hidden test, so it is removed
# after extraction; do not keep or re-fetch it for method development.

set -e

cd /home/user

RAW_DIR=data/downloads/eplb-openevolve
RAW="$RAW_DIR/expert-load.json"
URL="https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/071910dada7ce35e8b9b89b64ec1295f1b64590b/expert-load.json"
SHA256="d93580b3cb3e404a60237fa708271b360229a0c6fdfdd924a8ec4fdb440b71ad"

if [ ! -f data/eplb_train_windows.json ]; then
  mkdir -p "$RAW_DIR"
  if [ ! -f "$RAW" ]; then
    curl -sSL --retry 3 -o "$RAW" "$URL"
  fi
  echo "$SHA256  $RAW" | sha256sum -c -
  python data/prepare_eplb_windows.py --raw "$RAW" --output data/eplb_train_windows.json
  rm -f "$RAW"
fi
echo "data/eplb_train_windows.json ready"
