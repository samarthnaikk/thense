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
uv run python -m thense.cli bhavcopy --from 2020-01-01 --to 2020-12-31 --workers 8 --consolidate
uv run python -m thense.cli indices --period max
uv run python -m thense.cli hf --dataset daily     # or: minute | options

# or via the wrapper script
uv run python scripts/download_data.py bhavcopy --from 2024-01-01
```

Downloads are resumable: each trading day is one Parquet file and is skipped if
it already exists (`--force` to re-fetch). HTTP 404 simply means a holiday.

## Layout

```
src/thense/config.py            paths
src/thense/ingest/http.py       session, retries, polite pacing
src/thense/ingest/bhavcopy.py   NSE daily all-market bhavcopy
src/thense/ingest/security_master.py
src/thense/ingest/indices.py    Yahoo daily index/macro series
src/thense/ingest/hf_datasets.py
src/thense/cli.py               command-line entry point
scripts/download_data.py        wrapper
data/                           (gitignored) raw + processed Parquet
```

## Notes

- Data is for **internal research only**; NSE/broker data may not be redistributed.
- The HF daily dataset ends around 2026-01-22; recent days come from NSE bhavcopy.
