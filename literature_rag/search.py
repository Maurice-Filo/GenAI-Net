#!/usr/bin/env python3
"""Search the local open-access corpus; output is suitable for a Harness tool."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def fts_query(text: str) -> str:
    terms = [
        term for term in re.findall(r"[A-Za-z0-9]+", text)
        if term.upper() not in {"AND", "OR", "NOT"}
    ]
    if not terms:
        raise ValueError("query must contain at least one word or number")
    return " OR ".join(f'"{term}"' for term in terms)


def search_database(
    database: Path, query: str, *, limit: int = 8, topic: str | None = None
) -> list[dict[str, Any]]:
    """Return ranked passages from a read-only SQLite connection."""
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    topic_join = "JOIN paper_topics t ON t.pmcid = p.pmcid" if topic else ""
    topic_filter = "AND t.topic = ?" if topic else ""
    sql = f"""
        SELECT p.pmcid, p.title, p.doi, p.authors, p.publication_date, p.journal,
               p.source_url, snippet(paper_text, 2, '[', ']', ' ... ', 32) AS passage,
               bm25(paper_text, 8.0, 1.0, 2.0) AS score
        FROM paper_text
        JOIN papers p ON p.pmcid = paper_text.pmcid
        {topic_join}
        WHERE paper_text MATCH ? {topic_filter}
        ORDER BY score
        LIMIT ?
    """
    params: list[object] = [fts_query(query)]
    if topic:
        params.append(topic)
    params.append(max(1, min(int(limit), 50)))
    try:
        return [dict(row) for row in connection.execute(sql, params)]
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--topic")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--database", type=Path, default=Path(__file__).parent / "index" / "literature.sqlite3"
    )
    args = parser.parse_args()
    if not args.database.exists():
        parser.error(f"index does not exist: {args.database}; run fetch_papers.py first")

    rows = search_database(
        args.database, args.query, limit=args.limit, topic=args.topic
    )

    if args.json:
        print(json.dumps({"query": args.query, "results": rows}, indent=2, ensure_ascii=True))
    else:
        for index, row in enumerate(rows, 1):
            identifier = row["doi"] or row["pmcid"]
            print(f"{index}. {row['title']} ({identifier})")
            print(f"   {row['passage']}")
            print(f"   {row['source_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
