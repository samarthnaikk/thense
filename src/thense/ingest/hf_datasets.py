"""Download free bulk NSE datasets from the Hugging Face Hub.

Two datasets cover the whole market at zero cost:

* indian-stock-market-minute-data  (2,500+ symbols; 1-min 2022-2026, daily 2000-2026)
* nse-options-intraday             (NIFTY/BANKNIFTY/SENSEX options)

Both are Parquet and can be streamed. We snapshot the Parquet shards locally.
"""

from __future__ import annotations

from pathlib import Path

from ..config import HF_DIR, ensure_dirs

MINUTE_REPO = "xxparthparekhxx/indian-stock-market-minute-data"
OPTIONS_REPO = "rissin/nse-options-intraday"

# pattern sets to keep downloads focused
PATTERNS = {
    "daily": ["README.md", ".gitattributes", "day/*.parquet"],
    "minute": ["README.md", ".gitattributes", "minute/*.parquet"],
    "minute-data": ["README.md", ".gitattributes", "minute/*.parquet"],
    "options": ["README.md", ".gitattributes", "**/*.parquet"],
}


def download(dataset: str, max_workers: int = 4) -> Path:
    """Snapshot a dataset (or a subset) into data/raw/hf/<repo>.

    ``max_workers`` controls parallel file downloads; keep it low (2-4) on
    flaky links, where high concurrency causes stalled transfers.
    """
    from huggingface_hub import snapshot_download

    ensure_dirs()
    if dataset == "daily":
        repo_id, patterns = MINUTE_REPO, PATTERNS["daily"]
    elif dataset == "minute":
        repo_id, patterns = MINUTE_REPO, PATTERNS["minute"]
    elif dataset == "options":
        repo_id, patterns = OPTIONS_REPO, PATTERNS["options"]
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    local = HF_DIR / repo_id.replace("/", "__")
    print(f"Downloading {repo_id} ({dataset}) -> {local}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local,
        allow_patterns=patterns,
        max_workers=max_workers,
    )
    print(f"  done: {local}")
    return local
