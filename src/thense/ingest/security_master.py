"""Download the NSE listed-securities master (EQUITY_L.csv)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..config import REFERENCE_DIR, ensure_dirs
from .http import get_bytes

EQUITY_L_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def download() -> Path:
    """Download EQUITY_L.csv and write both the raw CSV and a Parquet copy."""
    ensure_dirs()
    content = get_bytes(EQUITY_L_URL)
    if content is None:
        raise RuntimeError(f"security master not found at {EQUITY_L_URL}")

    csv_path = REFERENCE_DIR / "EQUITY_L.csv"
    parquet_path = REFERENCE_DIR / "security_master.parquet"
    csv_path.write_bytes(content)

    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip() for c in df.columns]
    df.to_parquet(parquet_path, index=False)
    print(f"Security master: {len(df):,} rows -> {parquet_path}")
    return parquet_path
