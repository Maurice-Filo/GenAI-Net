#!/usr/bin/env python3
"""Serve read-only literature searches through an auditable file queue."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from search import search_database


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--database", type=Path, default=Path(__file__).parent / "index" / "literature.sqlite3"
    )
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if not args.database.exists():
        parser.error(f"index does not exist: {args.database}")
    args.queue.mkdir(parents=True, exist_ok=True)
    os.chmod(args.queue, 0o700)

    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"literature queue ready: {args.queue.resolve()}", flush=True)
    while not stopping:
        handled = False
        for request_path in sorted(args.queue.glob("*.request.json")):
            response_path = request_path.with_name(
                request_path.name.replace(".request.json", ".response.json")
            )
            if response_path.exists():
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                query = str(request["query"])
                results = search_database(
                    args.database,
                    query,
                    limit=int(request.get("limit", 8)),
                    topic=request.get("topic"),
                )
                response = {"ok": True, "query": query, "results": results}
            except json.JSONDecodeError:
                continue
            except Exception as error:
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            write_json(response_path, response)
            handled = True
        if not handled:
            time.sleep(max(0.02, args.poll_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
