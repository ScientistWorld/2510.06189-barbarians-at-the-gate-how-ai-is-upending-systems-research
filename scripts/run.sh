#!/bin/bash
# End-to-end reproduction of the reference: publish baselines + method,
# score them on the visible and hidden windows, then validate the workspace.
# (The hardening audit is run separately: python redteam/hardening/run_audit.py)
set -e
cd /home/user
bash scripts/reproduce.sh
python validate.py --compare
