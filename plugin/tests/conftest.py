"""Tests of the turbo-socialsci engine on public development inputs only (run from the repository root
with solutions/.venv/bin/python -m pytest plugin/tests)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "plugin/turbo-socialsci/engine/src"
for p in (str(ROOT), str(ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)
