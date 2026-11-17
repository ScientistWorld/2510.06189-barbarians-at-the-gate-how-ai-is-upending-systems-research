#!/bin/bash
# End-to-end reproduction of the reference: publish baselines + method,
# score them on the visible and hidden pairs, validate the workspace, then
# run the hardening audit (every variant and shortcut through the hidden test).
set -e
cd /home/user
echo "== environment"
python --version
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
grep -m1 "model name" /proc/cpuinfo || true
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
python -c "import ctypes; print('landlock abi', ctypes.CDLL(None).syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint32(1)))"
echo "== reproduce"
bash scripts/reproduce.sh
echo "== validate"
python validate.py --compare || echo "validate exit $?"
echo "== audit"
python redteam/hardening/run_audit.py
