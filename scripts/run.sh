#!/bin/bash
# Hardening job: reproduce the reference end to end, validate, then run the
# hardening audit (every variant and shortcut through the hidden test).
set -u
cd /home/user
mkdir -p redteam/hardening/results redteam/hardening/.work/tmp
export TMPDIR=/home/user/redteam/hardening/.work/tmp
{
  echo "== environment"
  python --version
  python -c "import sys, torch; print('torch', torch.__version__, '| sys.monitoring', hasattr(sys, 'monitoring'))"
  echo "== reproduce"
  bash scripts/reproduce.sh; echo "reproduce exit $?"
  echo "== validate"
  python validate.py; echo "validate exit $?"
  echo "== audit"
  python redteam/hardening/run_audit.py; echo "audit exit $?"
} 2>&1 | tee redteam/hardening/results/job_log.txt
