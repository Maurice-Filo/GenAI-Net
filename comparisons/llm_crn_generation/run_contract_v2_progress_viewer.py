#!/usr/bin/env python3
"""Wait for a sentinel database, then serve it with campaign progress."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL4CRN.utils.results_viewer import serve_results_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", required=True)
    parser.add_argument("--wait-seconds", type=float, default=43200.0)
    args = parser.parse_args()

    database = args.database.expanduser().resolve()
    deadline = time.monotonic() + args.wait_seconds
    while not database.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for sentinel database: {database}")
        time.sleep(2.0)
    serve_results_database(
        database,
        host=args.host,
        port=args.port,
        access_token=args.token,
        campaign_plan=args.campaign_plan,
    )


if __name__ == "__main__":
    main()
