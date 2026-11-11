#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py run shortcut_omit_method_from_roster
