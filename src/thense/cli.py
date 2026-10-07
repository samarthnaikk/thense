"""Command-line entry point: `thense <command>` or `python -m thense.cli`."""

from __future__ import annotations

import argparse
from datetime import date

from . import __version__
from .config import ensure_dirs
from .ingest import bhavcopy, hf_datasets, indices, security_master
from .report import status


def _add_bhavcopy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--from", dest="start", default="2000-01-01",
                   help="start date YYYY-MM-DD (default 2000-01-01)")
    p.add_argument("--to", dest="end", default=date.today().isoformat(),
                   help="end date YYYY-MM-DD (default today)")
    p.add_argument("--workers", type=int, default=2, help="parallel downloads")
    p.add_argument("--delay", type=float, default=0.5,
                   help="min seconds between requests (WAF safety; default 0.5)")
    p.add_argument("--force", action="store_true", help="re-download existing days")
    p.add_argument("--consolidate", action="store_true",
                   help="merge per-day files into one Parquet after download")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thense", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("security-master", help="download NSE EQUITY_L.csv master")
    sub.add_parser("status", help="report local data coverage")

    p_bhav = sub.add_parser("bhavcopy", help="download daily all-market bhavcopy")
    _add_bhavcopy_args(p_bhav)

    p_idx = sub.add_parser("indices", help="download daily index/macro series")
    p_idx.add_argument("--period", default="max", help="yfinance period, e.g. max")

    p_hf = sub.add_parser("hf", help="download free bulk Hugging Face datasets")
    p_hf.add_argument("--dataset", choices=["daily", "minute", "options"],
                      default="daily")

    p_all = sub.add_parser("all", help="run the full acquisition pipeline")
    _add_bhavcopy_args(p_all)
    p_all.add_argument("--hf-datasets", nargs="*",
                       default=["daily"],
                       choices=["daily", "minute", "options"])
    p_all.add_argument("--skip-indices", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dirs()

    if args.command == "status":
        status()
    elif args.command == "security-master":
        security_master.download()
    elif args.command == "bhavcopy":
        stats = bhavcopy.download_range(
            bhavcopy.parse_date(args.start),
            bhavcopy.parse_date(args.end),
            workers=args.workers,
            delay=args.delay,
            force=args.force,
        )
        print("bhavcopy:", stats)
        if args.consolidate:
            bhavcopy.consolidate()
    elif args.command == "indices":
        indices.download(period=args.period)
    elif args.command == "hf":
        hf_datasets.download(args.dataset)
    elif args.command == "all":
        print("== security master ==")
        security_master.download()
        if not args.skip_indices:
            print("== indices / macro ==")
            indices.download(period="max")
        for ds in args.hf_datasets:
            print(f"== hf: {ds} ==")
            hf_datasets.download(ds)
        print("== bhavcopy ==")
        stats = bhavcopy.download_range(
            bhavcopy.parse_date(args.start),
            bhavcopy.parse_date(args.end),
            workers=args.workers,
            delay=args.delay,
            force=args.force,
        )
        print("bhavcopy:", stats)
        if args.consolidate:
            bhavcopy.consolidate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
