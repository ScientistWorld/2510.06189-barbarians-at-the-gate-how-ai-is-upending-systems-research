#!/usr/bin/env python3
"""shortcut_malformed_artifact: independently deployable audit policy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import policy
policy('shortcut_malformed_artifact')
