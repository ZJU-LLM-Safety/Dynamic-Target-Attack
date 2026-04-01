"""Thin wrapper so the harmfulness CLI can be called as a script."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.harmfulness.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
