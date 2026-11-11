#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py run brute_force_enumeration
