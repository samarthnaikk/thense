"""Shared HTTP helpers: browser-like session, retries and polite pacing."""

from __future__ import annotations

import random
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

_local = threading.local()


def get_session() -> requests.Session:
    """Return a per-thread requests Session with retries configured."""
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        retry = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _local.session = session
    return session


def polite_sleep(base: float = 0.15, jitter: float = 0.2) -> None:
    """Sleep a small randomized amount to avoid hammering the origin."""
    time.sleep(base + random.random() * jitter)


def get_bytes(url: str, *, timeout: int = 40) -> bytes | None:
    """GET a URL and return its bytes, or None on a definitive 404.

    Raises for other unexpected statuses so callers can distinguish a
    missing trading day (404) from a transient failure.
    """
    session = get_session()
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def get_text(url: str, *, timeout: int = 40) -> str | None:
    data = get_bytes(url, timeout=timeout)
    return None if data is None else data.decode("utf-8", errors="replace")
