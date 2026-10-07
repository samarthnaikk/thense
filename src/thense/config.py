"""Project paths and shared configuration."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

# Raw NSE capital-market bhavcopy, one parquet per trading day:
#   data/raw/bhavcopy/<YYYY>/<YYYY-MM-DD>.parquet
BHAVCOPY_RAW_DIR = RAW_DIR / "bhavcopy"
# Consolidated long-history bhavcopy:
BHAVCOPY_CONSOLIDATED = PROCESSED_DIR / "bhavcopy" / "bhavcopy.parquet"

# Daily index/vol series downloaded from Yahoo Finance:
INDICES_DIR = RAW_DIR / "indices"

# Hugging Face bulk datasets:
HF_DIR = RAW_DIR / "hf"

ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    REFERENCE_DIR,
    BHAVCOPY_RAW_DIR,
    INDICES_DIR,
    HF_DIR,
]


def ensure_dirs() -> None:
    """Create the data directory tree if missing."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
