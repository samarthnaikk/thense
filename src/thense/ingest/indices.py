"""Download daily index / macro series via Yahoo Finance.

Used for India VIX, Nifty/Bank Nifty benchmarks and overnight global cues.
Yahoo intraday is limited, so this module fetches *daily* history.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import INDICES_DIR, ensure_dirs

# name -> Yahoo ticker
SERIES = {
    "india_vix": "^INDIAVIX",
    "nifty50": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dxy": "DX-Y.NYB",
    "usdinr": "USDINR=X",
    "brent": "BZ=F",
    "gold": "GC=F",
}


def download(period: str = "max") -> dict[str, Path]:
    """Download each series to its own Parquet file."""
    import yfinance as yf

    ensure_dirs()
    written: dict[str, Path] = {}
    for name, ticker in SERIES.items():
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name} ({ticker}) failed: {exc}")
            continue
        if df is None or df.empty:
            print(f"  ! {name} ({ticker}) returned no data")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        path = INDICES_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written[name] = path
        print(f"  {name:10s} {ticker:12s} {len(df):>6,} rows -> {path.name}")
    return written
