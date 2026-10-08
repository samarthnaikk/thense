# thense

NSE market-data acquisition and ML forecasting research.

The full research dossier (data sources, competitors, feasibility, costs) lives in
`~/research/nse-intraday-ml-forecasting.md`. This repo implements **Phase 0/1**:
free, full-market NSE data acquisition.

## What it gets (all free)

| Source | Coverage | Output |
|---|---|---|
| NSE bhavcopy (official) | daily EOD, all securities, ~2000→today, both archive formats | `data/raw/bhavcopy/<year>/<date>.parquet` + consolidated |
| NSE EQUITY_L.csv | listed-security master | `data/reference/` |
| Yahoo Finance | India VIX, Nifty, Bank Nifty, Sensex, S&P, Nasdaq, DXY, USDINR, Brent, Gold (daily) | `data/raw/indices/*.parquet` |
| HF `indian-stock-market-minute-data` | 2,500+ symbols; daily 2000–2026, 1-min 2022–2026 | `data/raw/hf/...` |
| HF `nse-options-intraday` | NIFTY/BANKNIFTY/SENSEX options | `data/raw/hf/...` |

## Setup

Requires [uv](https://docs.astral.sh/uv/) (pins Python 3.12; the repo system Python is 3.14).

```bash
uv sync
```

## Usage

```bash
# security master + indices + HF daily + bhavcopy 2000→today, then consolidate
uv run python -m thense.cli all --from 2000-01-01 --consolidate

# individual steps
uv run python -m thense.cli security-master
uv run python -m thense.cli bhavcopy --from 2020-01-01 --to 2020-12-31 --workers 2 --delay 0.5 --consolidate
uv run python -m thense.cli indices --period max
uv run python -m thense.cli hf --dataset daily     # or: minute | options

# or via the wrapper script
uv run python scripts/download_data.py bhavcopy --from 2024-01-01

# long runs should be detached (survives the terminal):
python scripts/run_bg.py data/logs/bhavcopy_full.log -- \
    bhavcopy --from 2000-01-01 --workers 2 --delay 0.3 --consolidate
```

Downloads are resumable: each trading day is one Parquet file and is skipped if
it already exists (`--force` to re-fetch). HTTP 404 simply means a holiday.
Requests are globally rate-limited (`--delay`) to stay under NSE's WAF, and the
run stops early (still resumable) if NSE starts returning sustained 403s.

### Bhavcopy archive eras

| Era | Format | URL pattern |
|---|---|---|
| 2000 – 2023 | legacy | `content/historical/EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip` |
| 2024 – today | UDiFF | `content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip` |

Pre‑2010 files have a thinner schema (no `ISIN`/`TOTALTRADES`); those columns come
out null. All rows are normalised to one schema and concatenated into
`data/processed/bhavcopy/bhavcopy.parquet`.

## Layout

```
src/thense/config.py            paths
src/thense/ingest/http.py       session, retries, polite pacing
src/thense/ingest/bhavcopy.py   NSE daily all-market bhavcopy
src/thense/ingest/security_master.py
src/thense/ingest/indices.py    Yahoo daily index/macro series
src/thense/ingest/hf_datasets.py
src/thense/report.py            `thense status` coverage report
src/thense/cli.py               command-line entry point
scripts/download_data.py        wrapper
scripts/run_bg.py               detached launcher for long downloads
data/                           (gitignored) raw + processed Parquet
```

## Current coverage (as of 2026-10-07)

| Dataset | Coverage | Size |
|---|---|---|
| NSE bhavcopy | 6,625 trading days, **10,672,269 rows**, 8,897 symbols, 2000-01-03 → 2026-10-07 | 723 MB (6,625 files) |
| Security master | 2,599 listed securities | 0.3 MB |
| Indices / macro | India VIX, Nifty, Bank Nifty, Sensex + global cues (daily) | 3 MB |
| HF daily (Upstox-derived) | 2,535 symbols, 2000 → 2026-01-22, 6.97M rows | 193 MB |
| HF minute (Upstox-derived) | 2,535 symbols, **1-min 2022-01-03 → 2026-01-21, 713.45M rows** | 9.7 GB (8 shards) |
| HF options | incomplete (partial; use `hf-fetch --dataset options`) | ~112 MB |

Run `thense status` for the live picture.

## Notes

- Data is for **internal research only**; NSE/broker data may not be redistributed.
- The HF daily dataset is community-sourced (Upstox), stops ~2026-01-22, and is
  likely survivorship-biased; NSE bhavcopy is the official, point-in-time record
  (includes delisted names) and is current. Use bhavcopy as the source of truth.
- Bhavcopy has no delivery-% — add `sec_bhavdata_full_<DDMMYYYY>.csv` later if needed.
