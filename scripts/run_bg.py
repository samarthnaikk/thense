#!/usr/bin/env python
"""Launch a `thense` CLI command fully detached from the invoking shell.

The download is long-running and must survive the terminal/tool that started
it. This starts the CLI in a new session with stdio redirected to a log file,
so the parent can exit immediately.

Usage:
    python scripts/run_bg.py <logfile> -- <thense args...>

Example:
    python scripts/run_bg.py data/logs/bhavcopy_full.log -- \
        bhavcopy --from 2000-01-01 --workers 2 --delay 0.3 --consolidate
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print(__doc__)
        return 2
    split = argv.index("--")
    log_arg = argv[:split]
    cli_args = argv[split + 1 :]
    if not cli_args:
        print("error: no thense arguments after '--'")
        return 2

    root = Path(__file__).resolve().parents[1]
    log_path = Path(log_arg[0]) if log_arg else root / "data/logs/bg.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "thense.cli", *cli_args]

    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    print(f"started pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
