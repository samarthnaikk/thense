"""Download and normalise NSE capital-market bhavcopy (daily, all securities).

NSE publishes two archive formats over time:

* Old (pre-2024):  content/historical/EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip
* New (UDiFF):     content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip

Each trading day is written as one Parquet file so the download is resumable.
A 404 simply means "no file that day" (holiday/weekend) and is skipped.
"""

from __future__ import annotations

import io
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import BHAVCOPY_CONSOLIDATED, BHAVCOPY_RAW_DIR, ensure_dirs
from .http import get_bytes, polite_sleep, set_min_interval

# Set when NSE starts returning sustained 403s; pending work then short-circuits.
_stop = threading.Event()
_BLOCK_THRESHOLD = 20

ARCHIVE_BASE = "https://nsearchives.nseindia.com"

NEW_URL = ARCHIVE_BASE + "/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
OLD_URL = (
    ARCHIVE_BASE
    + "/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd}{mon}{yyyy}bhav.csv.zip"
)

CANONICAL_COLUMNS = [
    "date",
    "symbol",
    "series",
    "isin",
    "open",
    "high",
    "low",
    "close",
    "last",
    "prev_close",
    "volume",
    "turnover",
    "trades",
    "source_format",
]

_NEW_RENAME = {
    "TradDt": "date",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "ISIN": "isin",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}

_OLD_RENAME = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LAST": "last",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TIMESTAMP": "date",
    "TOTALTRADES": "trades",
    "ISIN": "isin",
}


# NSE's UDiFF (new-format) archive starts here; earlier dates only exist in
# the legacy per-month path. Trying the wrong format first costs a wasted
# request per day, so order by era and keep the other as a fallback.
NEW_FORMAT_START = date(2024, 1, 1)


def _urls_for(day: date) -> list[str]:
    ymd = day.strftime("%Y%m%d")
    mon = day.strftime("%b").upper()
    new = NEW_URL.format(ymd=ymd)
    old = OLD_URL.format(yyyy=day.year, mon=mon, dd=day.strftime("%d"))
    return [new, old] if day >= NEW_FORMAT_START else [old, new]


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            return pd.read_csv(fh)


def _normalise(raw: pd.DataFrame, day: date, source_format: str) -> pd.DataFrame:
    if source_format == "new":
        raw = raw[raw.get("FinInstrmTp", "STK") == "STK"].copy()
        df = raw.rename(columns=_NEW_RENAME)
    else:
        df = raw.rename(columns=_OLD_RENAME)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["source_format"] = source_format

    # Older archives omit columns (e.g. pre-2010 files have no ISIN or
    # TOTALTRADES), so materialise every canonical column before casting.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ("open", "high", "low", "close", "last", "prev_close", "turnover"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("volume", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df[CANONICAL_COLUMNS]
    df["symbol"] = df["symbol"].astype("string").str.strip()
    df["series"] = df["series"].astype("string").str.strip()
    df["isin"] = df["isin"].astype("string").str.strip()
    return df


def fetch_day(day: date) -> pd.DataFrame | None:
    """Fetch and normalise one day's bhavcopy, or None if not a trading day.

    Older dates may return 403 on the new-format path (not 404), so any
    failure on one URL falls through to the next URL before giving up.
    """
    for url in _urls_for(day):
        source_format = "new" if "BhavCopy_NSE_CM" in url else "old"
        try:
            content = get_bytes(url)
        except Exception:  # noqa: BLE001 — 403/5xx on a probe URL: try the other
            continue
        if content is None:
            continue
        try:
            raw = _read_zip_csv(content)
        except (zipfile.BadZipFile, IndexError, pd.errors.ParserError):
            continue
        return _normalise(raw, day, source_format)
    return None


def _day_path(day: date) -> Path:
    return BHAVCOPY_RAW_DIR / f"{day.year}" / f"{day.isoformat()}.parquet"


def download_day(day: date, *, force: bool = False) -> Path | None:
    """Download one day to Parquet. Returns the path (or None if no data)."""
    if _stop.is_set():
        return None
    path = _day_path(day)
    if path.exists() and not force:
        return path
    df = fetch_day(day)
    if df is None or df.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def download_range(
    start: date,
    end: date,
    *,
    workers: int = 2,
    delay: float = 0.5,
    force: bool = False,
) -> dict[str, int]:
    """Download bhavcopy for every weekday in [start, end] (resumable).

    ``delay`` is the global minimum spacing between outbound requests; keep it
    high enough to stay under NSE's WAF radar. The run stops early (and stays
    resumable) if NSE starts returning sustained 403s.
    """
    ensure_dirs()
    set_min_interval(delay)
    _stop.clear()
    days = list(_weekdays(start, end))
    stats = {
        "trading_days": 0,
        "skipped_existing": 0,
        "holidays": 0,
        "errors": 0,
        "blocked": 0,
    }

    pending = []
    for d in days:
        if _day_path(d).exists() and not force:
            stats["skipped_existing"] += 1
        else:
            pending.append(d)

    if not pending:
        print(f"Nothing to do: all {stats['skipped_existing']} days already present.")
        return stats

    consecutive_block = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_day, d, force=force): d for d in pending}
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="bhavcopy"):
                day = futures[fut]
                if _stop.is_set():
                    stats["blocked"] += 1
                    continue
                try:
                    path = fut.result()
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    consecutive_block = (
                        consecutive_block + 1 if "403" in str(exc) else 0
                    )
                    tqdm.write(f"  ! {day} error: {exc}")
                    if consecutive_block >= _BLOCK_THRESHOLD:
                        _stop.set()
                        tqdm.write(
                            "  !! sustained 403s — NSE is likely blocking this IP. "
                            "Stopping; re-run later (download is resumable)."
                        )
                    continue
                consecutive_block = 0
                if path is None:
                    stats["holidays"] += 1
                else:
                    stats["trading_days"] += 1
        finally:
            if _stop.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
    return stats


def consolidate() -> Path:
    """Merge all per-day Parquet files into one long-history Parquet."""
    import duckdb

    ensure_dirs()
    pattern = str(BHAVCOPY_RAW_DIR / "*" / "*.parquet")
    BHAVCOPY_CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{pattern}', union_by_name=true)
            ORDER BY date, symbol
        ) TO '{BHAVCOPY_CONSOLIDATED}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{BHAVCOPY_CONSOLIDATED}')"
    ).fetchone()[0]
    con.close()
    print(f"Consolidated {n:,} rows -> {BHAVCOPY_CONSOLIDATED}")
    return BHAVCOPY_CONSOLIDATED


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
