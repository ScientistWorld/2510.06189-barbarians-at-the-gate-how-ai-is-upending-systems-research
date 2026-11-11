#!/usr/bin/env python3
"""weaker_without_snake: independently deployable audit policy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import policy
policy('weaker_without_snake')
