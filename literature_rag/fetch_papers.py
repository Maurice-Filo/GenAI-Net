#!/usr/bin/env python3
"""Download a balanced, open-access Europe PMC corpus and build SQLite FTS."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


API_ROOT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DEFAULT_USER_AGENT = "GenAI-Net literature pilot/1.0 (open-access research corpus)"


@dataclass(frozen=True)
class Candidate:
    pmcid: str
    title: str
    doi: str
    authors: str
    publication_date: str
    journal: str
    topic: str


def request_bytes(url: str, *, user_agent: str, retries: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed after {retries} attempts: {url}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def search_candidates(
    topic: str, query: str, *, user_agent: str, page_size: int = 100
) -> list[Candidate]:
    # IN_PMC guarantees a stable JATS full-text endpoint; OPEN_ACCESS limits the
    # local corpus to articles Europe PMC explicitly marks as open access.
    full_query = f"OPEN_ACCESS:Y AND IN_PMC:Y AND ({query})"
    params = urllib.parse.urlencode(
        {"query": full_query, "format": "json", "resultType": "core", "pageSize": page_size}
    )
    payload = json.loads(request_bytes(f"{API_ROOT}/search?{params}", user_agent=user_agent))
    candidates: list[Candidate] = []
    for result in payload.get("resultList", {}).get("result", []):
        pmcid = str(result.get("pmcid") or "").upper()
        if not re.fullmatch(r"PMC\d+", pmcid):
            continue
        candidates.append(
            Candidate(
                pmcid=pmcid,
                title=str(result.get("title") or "Untitled").strip(),
                doi=str(result.get("doi") or "").strip(),
                authors=str(result.get("authorString") or "").strip(),
                publication_date=str(
                    result.get("firstPublicationDate") or result.get("journalInfo", {}).get("printPublicationDate") or ""
                ).strip(),
                journal=str(result.get("journalTitle") or "").strip(),
                topic=topic,
            )
        )
    return candidates


def local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def node_text(node: ET.Element) -> str:
    return html.unescape(" ".join("".join(node.itertext()).split()))


def extract_jats(xml_bytes: bytes) -> tuple[str, str]:
    root = ET.fromstring(xml_bytes)
    title = ""
    for node in root.iter():
        if local_name(node.tag) == "article-title":
            title = node_text(node)
            break

    sections: list[str] = []
    for node in root.iter():
        tag = local_name(node.tag)
        if tag == "abstract":
            text = node_text(node)
            if text:
                sections.append("ABSTRACT\n" + text)
        elif tag == "sec":
            direct_title = next(
                (node_text(child) for child in node if local_name(child.tag) == "title"), ""
            )
            paragraphs = [
                node_text(child)
                for child in node
                if local_name(child.tag) in {"p", "disp-quote"} and node_text(child)
            ]
            if paragraphs:
                sections.append("\n".join(filter(None, [direct_title.upper(), *paragraphs])))
    return title, "\n\n".join(sections)


def connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS papers (
            pmcid TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            doi TEXT,
            authors TEXT,
            publication_date TEXT,
            journal TEXT,
            source_url TEXT NOT NULL,
            license_status TEXT NOT NULL,
            xml_path TEXT NOT NULL,
            text_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            downloaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS paper_topics (
            pmcid TEXT NOT NULL,
            topic TEXT NOT NULL,
            PRIMARY KEY (pmcid, topic),
            FOREIGN KEY (pmcid) REFERENCES papers(pmcid)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS paper_text USING fts5(
            pmcid UNINDEXED, title, body, tokenize='porter unicode61'
        );
        """
    )
    return connection


def balanced_candidates(
    topics: dict[str, dict[str, object]], *, limit: int, user_agent: str
) -> Iterable[Candidate]:
    pools: dict[str, list[Candidate]] = {}
    for topic, config in topics.items():
        quota = min(int(config["quota"]), limit)
        pools[topic] = search_candidates(
            topic, str(config["query"]), user_agent=user_agent, page_size=min(1000, max(100, quota * 3))
        )
        print(f"{topic}: discovered {len(pools[topic])} candidates for quota {quota}", flush=True)

    emitted: set[str] = set()
    topic_counts = {topic: 0 for topic in topics}
    made_progress = True
    while len(emitted) < limit and made_progress:
        made_progress = False
        for topic, config in topics.items():
            if topic_counts[topic] >= int(config["quota"]):
                continue
            while pools[topic]:
                candidate = pools[topic].pop(0)
                if candidate.pmcid in emitted:
                    continue
                emitted.add(candidate.pmcid)
                topic_counts[topic] += 1
                made_progress = True
                yield candidate
                break
            if len(emitted) >= limit:
                break

    # Supply replacements for records whose full text is unavailable or cannot
    # be parsed. The main loop still stops exactly at the requested corpus size.
    made_progress = True
    while made_progress:
        made_progress = False
        for topic in topics:
            while pools[topic]:
                candidate = pools[topic].pop(0)
                if candidate.pmcid in emitted:
                    continue
                emitted.add(candidate.pmcid)
                made_progress = True
                yield candidate
                break


def download_candidate(
    candidate: Candidate, *, root: Path, connection: sqlite3.Connection, user_agent: str
) -> bool:
    xml_dir = root / "papers" / "xml"
    text_dir = root / "papers" / "text"
    xml_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    xml_path = xml_dir / f"{candidate.pmcid}.xml"
    text_path = text_dir / f"{candidate.pmcid}.txt"
    source_url = f"{API_ROOT}/{candidate.pmcid}/fullTextXML"
    try:
        xml_bytes = request_bytes(source_url, user_agent=user_agent)
        parsed_title, body = extract_jats(xml_bytes)
    except (RuntimeError, ET.ParseError) as error:
        print(f"skip {candidate.pmcid}: {error}", file=sys.stderr, flush=True)
        return False
    if len(body) < 1000:
        print(f"skip {candidate.pmcid}: extracted text is too short", file=sys.stderr, flush=True)
        return False

    title = parsed_title or candidate.title
    xml_path.write_bytes(xml_bytes)
    text_path.write_text(f"{title}\n\n{body}\n", encoding="utf-8")
    digest = hashlib.sha256(xml_bytes).hexdigest()
    with connection:
        connection.execute(
            """INSERT OR REPLACE INTO papers
               (pmcid, title, doi, authors, publication_date, journal, source_url,
                license_status, xml_path, text_path, sha256)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'Europe PMC OPEN_ACCESS:Y', ?, ?, ?)""",
            (
                candidate.pmcid,
                title,
                candidate.doi,
                candidate.authors,
                candidate.publication_date,
                candidate.journal,
                source_url,
                str(xml_path.relative_to(root)),
                str(text_path.relative_to(root)),
                digest,
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO paper_topics (pmcid, topic) VALUES (?, ?)",
            (candidate.pmcid, candidate.topic),
        )
        connection.execute("DELETE FROM paper_text WHERE pmcid = ?", (candidate.pmcid,))
        connection.execute(
            "INSERT INTO paper_text (pmcid, title, body) VALUES (?, ?, ?)",
            (candidate.pmcid, title, body),
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--topics", type=Path, default=Path(__file__).with_name("topics.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between full-text requests")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    topics = json.loads(args.topics.read_text(encoding="utf-8"))
    if sum(int(config["quota"]) for config in topics.values()) < args.limit:
        parser.error("topic quotas sum to less than --limit")
    root = args.root.resolve()
    connection = connect_index(root / "index" / "literature.sqlite3")
    existing = {row[0] for row in connection.execute("SELECT pmcid FROM papers")}
    downloaded = len(existing)
    print(f"corpus currently contains {downloaded} papers; target is {args.limit}", flush=True)
    for candidate in balanced_candidates(topics, limit=args.limit, user_agent=args.user_agent):
        if downloaded >= args.limit:
            break
        if candidate.pmcid in existing:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO paper_topics (pmcid, topic) VALUES (?, ?)",
                    (candidate.pmcid, candidate.topic),
                )
            continue
        if download_candidate(candidate, root=root, connection=connection, user_agent=args.user_agent):
            existing.add(candidate.pmcid)
            downloaded += 1
            print(
                f"[{downloaded}/{args.limit}] [{candidate.topic}] "
                f"{candidate.pmcid} {candidate.title}",
                flush=True,
            )
        time.sleep(args.delay)
    connection.close()
    print(f"complete: {downloaded} papers in {root}", flush=True)
    return 0 if downloaded >= args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
