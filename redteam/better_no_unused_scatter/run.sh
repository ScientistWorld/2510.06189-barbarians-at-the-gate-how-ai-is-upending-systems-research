#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py run better_no_unused_scatter
