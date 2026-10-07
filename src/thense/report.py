"""Report on locally acquired data coverage."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import (
    BHAVCOPY_RAW_DIR,
    HF_DIR,
    INDICES_DIR,
    REFERENCE_DIR,
)


def _dir_size(path: Path) -> str:
    if not path.exists():
        return "-"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.0f}PB"


def _parquet_glob(root: Path) -> str:
    return str(root / "**" / "*.parquet")


def status() -> None:
    con = duckdb.connect()
    print("=" * 72)
    print("thense data coverage")
    print("=" * 72)

    # --- bhavcopy (official daily, all securities) ---
    bhav_glob = _parquet_glob(BHAVCOPY_RAW_DIR)
    n_files = len(list(BHAVCOPY_RAW_DIR.rglob("*.parquet"))) if BHAVCOPY_RAW_DIR.exists() else 0
    print(f"\nNSE bhavcopy  (files: {n_files}, size: {_dir_size(BHAVCOPY_RAW_DIR)})")
    if n_files:
        row = con.execute(
            f"""SELECT count(*) AS n, count(DISTINCT symbol) AS syms,
                       min(date) AS dmin, max(date) AS dmax
                FROM read_parquet('{bhav_glob}', union_by_name=true)"""
        ).fetchone()
        print(f"  rows={row[0]:,}  symbols={row[1]:,}  range={row[2]} .. {row[3]}")

    # --- reference / security master ---
    sm = REFERENCE_DIR / "security_master.parquet"
    if sm.exists():
        n = con.execute(f"SELECT count(*) FROM read_parquet('{sm}')").fetchone()[0]
        print(f"\nSecurity master (EQUITY_L.csv): {n:,} securities")

    # --- index / macro series ---
    print(f"\nIndices / macro  (size: {_dir_size(INDICES_DIR)})")
    if INDICES_DIR.exists():
        for f in sorted(INDICES_DIR.glob("*.parquet")):
            try:
                r = con.execute(
                    f"""SELECT count(*) AS n, min(date) AS dmin, max(date) AS dmax
                        FROM read_parquet('{f}')"""
                ).fetchone()
                print(f"  {f.stem:10s} rows={r[0]:>7,}  {r[1]} .. {r[2]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {f.stem:10s} (unreadable: {exc})")

    # --- Hugging Face bulk datasets ---
    print(f"\nHugging Face bulk  (size: {_dir_size(HF_DIR)})")
    if HF_DIR.exists():
        for d in sorted(p for p in HF_DIR.iterdir() if p.is_dir()):
            files = sorted(d.rglob("*.parquet"))
            print(f"  {d.name}: {len(files)} parquet files, {_dir_size(d)}")

    con.close()
    print()
