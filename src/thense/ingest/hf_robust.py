"""Robust, resumable HTTP fetcher for Hugging Face Parquet shards.

The stock ``huggingface_hub`` downloader can hang indefinitely when the link
degrades mid-transfer. This fetches the dataset ``resolve`` URLs directly with
a socket read timeout, and resumes from the last received byte after every
stall or error (HTTP Range). Downloads land in ``data/raw/hf/<repo>/<path>``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ..config import HF_DIR, ensure_dirs

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
CHUNK = 1 << 20  # 1 MiB

REPOS = {
    "minute": ("xxparthparekhxx/indian-stock-market-minute-data", "minute/"),
    "daily": ("xxparthparekhxx/indian-stock-market-minute-data", "day/"),
    "options": ("rissin/nse-options-intraday", ""),
}


def _token() -> str | None:
    try:
        tok = (Path.home() / ".cache/huggingface/token").read_text().strip()
        return tok or None
    except OSError:
        return None


def _headers(pos: int) -> dict[str, str]:
    headers = {"User-Agent": UA}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if pos:
        headers["Range"] = f"bytes={pos}-"
    return headers


def _total_from_headers(resp: requests.Response) -> int | None:
    if resp.status_code == 206:
        content_range = resp.headers.get("Content-Range", "")
        if "/" in content_range:
            tail = content_range.rsplit("/", 1)[-1]
            return int(tail) if tail.isdigit() else None
    content_length = resp.headers.get("Content-Length")
    return int(content_length) if content_length and content_length.isdigit() else None


def list_files(repo: str, prefix: str) -> list[tuple[str, int]]:
    """List (rfilename, size) for Parquet files under ``prefix`` with retries."""
    from huggingface_hub import HfApi

    for attempt in range(6):
        try:
            info = HfApi().dataset_info(repo, files_metadata=True)
            files = [
                (s.rfilename, int(s.size or 0))
                for s in info.siblings
                if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet")
            ]
            if files:
                return sorted(files)
        except Exception:  # noqa: BLE001 — flaky API: retry
            pass
        time.sleep(min(20, 2**attempt))
    return []


def fetch_file(
    repo: str,
    rfilename: str,
    size: int,
    dest: Path,
    *,
    read_timeout: int = 30,
    max_retries: int = 500,
) -> bool:
    """Download one file to ``dest``, resuming on stalls. True when complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if size and dest.exists() and dest.stat().st_size >= size:
        return True

    part = dest.with_suffix(dest.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    total = size or None
    url = RESOLVE.format(repo=repo, path=rfilename)

    for attempt in range(max_retries):
        if total and pos >= total:
            part.replace(dest)
            return True
        try:
            with requests.get(
                url, headers=_headers(pos), stream=True,
                timeout=(15, read_timeout), allow_redirects=True,
            ) as resp:
                if resp.status_code == 416:  # already have it all
                    part.replace(dest)
                    return True
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                if resp.status_code == 200 and pos:
                    pos = 0  # server ignored Range: start over
                total = _total_from_headers(resp) or total
                mode = "ab" if pos else "wb"
                with open(part, mode) as fh:
                    for chunk in resp.iter_content(CHUNK):
                        if chunk:
                            fh.write(chunk)
                            pos += len(chunk)
        except Exception:  # noqa: BLE001 — stall/timeout: back off and resume
            time.sleep(min(30, 2 + attempt))
            continue

    return False


def fetch(dataset: str, workers: int = 4) -> None:
    """Download all Parquet shards for a dataset, resuming where possible."""
    ensure_dirs()
    repo, prefix = REPOS[dataset]
    base = HF_DIR / repo.replace("/", "__")
    files = list_files(repo, prefix)
    if not files:
        raise RuntimeError(f"could not list files for {repo} [{prefix or '*'}]")

    total_bytes = sum(s for _, s in files)
    print(
        f"hf-fetch: {repo} [{prefix or '*'}] -> {base} "
        f"({len(files)} files, {total_bytes / 1e9:.2f} GB)",
        flush=True,
    )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_file, repo, name, size, base / name): name
            for name, size in files
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  ! {name}: {exc}", flush=True)
            if ok:
                done += 1
                print(f"  ok {name}", flush=True)
    print(f"hf-fetch done: {done}/{len(files)} files", flush=True)
