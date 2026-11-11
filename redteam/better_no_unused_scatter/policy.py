#!/usr/bin/env python3
"""better_no_unused_scatter: independently deployable audit policy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import policy
policy('better_no_unused_scatter')
