#!/usr/bin/env python
"""Thin wrapper so data acquisition can be run as `python scripts/download_data.py`.

All logic lives in the `thense` package; this only forwards to its CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thense.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
