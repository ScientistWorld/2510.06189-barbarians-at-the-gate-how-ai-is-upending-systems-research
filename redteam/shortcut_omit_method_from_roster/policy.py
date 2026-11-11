#!/usr/bin/env python3
"""shortcut_omit_method_from_roster: independently deployable audit policy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import policy
policy('shortcut_omit_method_from_roster')
