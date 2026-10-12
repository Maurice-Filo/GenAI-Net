"""Transactional per-run evaluation cache, work ledger and event journal."""

from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import threading
import time
import zlib


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class BudgetExhausted(RuntimeError):
    pass


class Store:
    def __init__(self, path, budget, evaluator_identity):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations(key TEXT PRIMARY KEY,payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluation_summaries(key TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS diagnostics(key TEXT PRIMARY KEY,payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT,created REAL);
            CREATE TABLE IF NOT EXISTS visits(key TEXT PRIMARY KEY,n INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS processing(key TEXT PRIMARY KEY,status TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS designs(key TEXT PRIMARY KEY,payload TEXT);
        """)
        self.identity = evaluator_identity
        self.budget = budget
        self.inflight = {}
        for key, value in (("identity", evaluator_identity), ("budget", budget)):
            old = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if old is not None and json.loads(old[0]) != value:
                self.db.close()
                raise ValueError(f"Run store {key} mismatch")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, canonical(value)))

    def event(self, kind, payload):
        with self.lock:
            cursor = self.db.execute("INSERT INTO events(kind,payload,created) VALUES (?,?,?)",
                                     (kind, canonical(payload), time.time()))
            return cursor.lastrowid

    @contextmanager
    def transaction(self):
        """Group a logical operation without weakening FULL durability."""
        with self.lock:
            nested = self.db.in_transaction
            if not nested:
                self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                if not nested:
                    self.db.execute("COMMIT")
            except BaseException:
                if not nested:
                    self.db.execute("ROLLBACK")
                raise

    def count(self, key, increment=0):
        with self.lock:
            if increment:
                self.db.execute("INSERT INTO visits VALUES (?,?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n", (key, increment))
            row = self.db.execute("SELECT n FROM visits WHERE key=?", (key,)).fetchone()
            return row[0] if row else 0

    def counts(self, keys):
        keys = list(dict.fromkeys(keys))
        values = dict.fromkeys(keys, 0)
        with self.lock:
            for offset in range(0, len(keys), 500):
                batch = keys[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                values.update(self.db.execute(f"SELECT key,n FROM visits WHERE key IN ({placeholders})", batch))
        return values

    def evaluate(self, candidate, fn, *, category="search", include_diagnostics=True):
        # Float.hex prevents accidental rounding-based cache equivalence. Rates stay aligned to IDs.
        key = self.evaluation_key(candidate)
        exhausted = False
        with self.lock:
            with self.transaction():
                self.count("evaluation_attempts", 1)
                row = self._read(key, include_diagnostics)
                if row is not None:
                    self.count("cache_hits", 1)
                    waiting = None
                else:
                    waiting = self.inflight.get(key)
                if row is None and waiting is None:
                    if self.count("expensive_evaluations") >= self.budget:
                        exhausted = True
                    else:
                        self.count("expensive_evaluations", 1)
                        self.count(f"evaluations/{category}", 1)
                        self.event("evaluation_reserved", {"key": key, "category": category})
                elif waiting is not None:
                    self.count("coalesced_hits", 1)
            if row is None and waiting is None and not exhausted:
                self.inflight[key] = Future()
        if exhausted:
            raise BudgetExhausted("Evaluation budget exhausted")
        if row is not None:
            return self._decode(row, include_diagnostics), True
        if waiting is not None:
            waiting.result()
            return self.cached(candidate, include_diagnostics=include_diagnostics), True
        future = self.inflight[key]
        try:
            payload = fn(candidate)
            summary = {k: v for k, v in payload.items() if k != "task_info"}
            packed = zlib.compress(canonical(payload["task_info"]).encode()) if "task_info" in payload else None
            with self.transaction():
                self.db.execute("INSERT INTO evaluation_summaries VALUES (?,?)", (key, canonical(summary)))
                if packed is not None:
                    self.db.execute("INSERT INTO diagnostics VALUES (?,?)", (key, packed))
                self.event("evaluation_completed", {"key": key, "loss": payload["loss"]})
            future.set_result(None)
            return (payload if include_diagnostics else summary), False
        except BaseException as error:
            with self.lock:
                try:
                    self.event("evaluation_failed", {"key": key, "type": type(error).__name__, "message": str(error)})
                finally:
                    future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.inflight.pop(key, None)

    def processing(self, key, status, payload):
        with self.transaction():
            compact = dict(payload)
            if compact.get("best"):
                compact["best"] = {k: v for k, v in compact["best"].items() if k != "task_info"}
            self.db.execute("INSERT OR REPLACE INTO processing VALUES (?,?,?)", (key, status, canonical(compact)))
            self.event("processing", {"key": key, "status": status,
                                      **{k: v for k, v in compact.items() if k != "best"}})

    def processing_state(self, key):
        with self.lock:
            row = self.db.execute("SELECT status,payload FROM processing WHERE key=?", (key,)).fetchone()
            return (row[0], json.loads(row[1])) if row else None

    def save_design(self, record):
        with self.lock:
            compact = {k: v for k, v in record.items() if k != "task_info"}
            compact["evaluation_cache_key"] = self.evaluation_key(record)
            self.db.execute("INSERT OR REPLACE INTO designs VALUES (?,?)", (record["candidate_hash"], canonical(compact)))

    def evaluation_key(self, candidate):
        pairs = sorted((int(i), float(k).hex()) for i, k in zip(candidate["reaction_ids"], candidate["rates"]))
        return fingerprint({"evaluator": self.identity, "pairs": pairs})

    def _read(self, key, include_diagnostics):
        summary = self.db.execute("SELECT payload FROM evaluation_summaries WHERE key=?", (key,)).fetchone()
        if summary is not None:
            info = self.db.execute("SELECT payload FROM diagnostics WHERE key=?", (key,)).fetchone() if include_diagnostics else None
            return (summary[0], info[0] if info else None, False)
        # Read existing monolithic caches without rewriting or invalidating them.
        legacy = self.db.execute("SELECT payload FROM evaluations WHERE key=?", (key,)).fetchone()
        return (legacy[0], None, True) if legacy else None

    @staticmethod
    def _decode(row, include_diagnostics):
        payload, info, legacy = row
        result = json.loads(zlib.decompress(payload)) if legacy else json.loads(payload)
        if info is not None:
            result["task_info"] = json.loads(zlib.decompress(info))
        if not include_diagnostics:
            result.pop("task_info", None)
        return result

    def cached(self, candidate, *, include_diagnostics=True):
        with self.lock:
            row = self._read(self.evaluation_key(candidate), include_diagnostics)
            if row is None:
                raise KeyError("Evaluation cache record missing")
        return self._decode(row, include_diagnostics)

    def counters(self):
        with self.lock:
            return dict(self.db.execute("SELECT key,n FROM visits WHERE key NOT LIKE 'region:%' AND key NOT LIKE 'topology:%'"))

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        self.db.close()
