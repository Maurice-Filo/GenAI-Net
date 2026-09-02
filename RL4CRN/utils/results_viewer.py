"""Small read-only web viewer for RL4CRN results databases."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import secrets
import sqlite3
import statistics
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse


class CampaignProgressReader:
    """Combine a declarative paper plan with read-only launcher status artifacts."""

    def __init__(self, plan_path: str | Path) -> None:
        self.path = Path(plan_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def snapshot(self) -> Dict[str, Any]:
        plan = json.loads(self.path.read_text(encoding="utf-8"))
        campaigns = [self._campaign(item) for item in plan.get("campaigns", [])]
        totals = {
            "runs": sum(item["total_runs"] for item in campaigns),
            "completed": sum(item["completed"] for item in campaigns),
            "active": sum(item["active"] for item in campaigns),
            "pending": sum(item["pending"] for item in campaigns),
            "failed": sum(item["failed"] for item in campaigns),
        }
        totals["progress_percent"] = (
            100.0 * sum(item["progress_units"] for item in campaigns)
            / max(1, sum(item["total_units"] for item in campaigns))
        )
        return {
            "title": plan.get("title", "Paper experiment plan"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "totals": totals,
            "campaigns": campaigns,
        }

    def _campaign(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(item)
        status_path = Path(item["status_path"]).expanduser() if item.get("status_path") else None
        manifest: Dict[str, Any] = {}
        manifest_path = status_path.parent / "campaign_manifest.json" if status_path else None
        if manifest_path and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
        use_manifest_scope = bool(item.get("conditional") and manifest)
        tasks = list(
            manifest.get("tasks", item.get("tasks", []))
            if use_manifest_scope
            else item.get("tasks", [])
        )
        manifest_seeds = manifest.get("seeds")
        seeds = (
            len(manifest_seeds)
            if use_manifest_scope and isinstance(manifest_seeds, list)
            else int(item.get("seeds", 0))
        )
        epochs = int(item.get("epochs", 0))
        total_runs = len(tasks) * seeds
        potential_runs = total_runs
        if item.get("conditional") and not manifest:
            total_runs = 0
        status: Dict[str, Any] = {}
        if status_path and status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = {}
        task_set = set(tasks)
        selected = lambda rows: [row for row in rows if row.get("task") in task_set]
        completed = len(selected(status.get("completed", [])))
        active_rows = selected(status.get("active", []))
        failed = len(selected(status.get("failed", [])))
        pending = len(selected(status.get("pending", [])))
        if not status:
            pending = total_runs
        active_progress = [self._active_progress(status_path, row, epochs) for row in active_rows]
        progress_units = completed * epochs + sum(row["epoch"] for row in active_progress)
        total_units = total_runs * epochs
        rates = [
            row["elapsed_seconds"] / row["epoch"]
            for row in active_progress
            if row.get("epoch", 0) > 0 and row.get("elapsed_seconds", 0) > 0
        ]
        concurrency = max(1, int(manifest.get("max_parallel", len(active_rows) or 1)))
        eta_seconds = None
        if rates and progress_units < total_units:
            eta_seconds = statistics.median(rates) * (total_units - progress_units) / concurrency
        phase = str(item.get("phase", "scheduled"))
        if failed:
            phase = "attention"
        elif total_runs and completed >= total_runs:
            phase = "completed"
        elif active_rows:
            phase = "running"
        result.update(
            {
                "total_runs": total_runs,
                "potential_runs": potential_runs,
                "tasks": tasks,
                "seeds": seeds,
                "completed": completed,
                "active": len(active_rows),
                "pending": pending,
                "failed": failed,
                "phase": phase,
                "active_runs": active_progress,
                "progress_units": progress_units,
                "total_units": total_units,
                "progress_percent": 100.0 * progress_units / max(1, total_units),
                "status_updated_at": status.get("updated_at"),
                "eta_seconds": eta_seconds,
            }
        )
        return result

    @staticmethod
    def _active_progress(status_path: Optional[Path], row: Dict[str, Any], epochs: int) -> Dict[str, Any]:
        progress = dict(row)
        progress["epoch"] = 0
        progress["target_epoch"] = epochs
        if status_path is None:
            return progress
        root = status_path.parent / "runs"
        pattern = f"*/{row.get('task', '*')}_*seed{row.get('seed', '*')}_*/progress.csv"
        matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            return progress
        try:
            with matches[0].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                progress["epoch"] = min(epochs, int(float(rows[-1].get("step", 0))))
                progress["elapsed_seconds"] = float(rows[-1].get("elapsed_seconds", 0))
                progress["best_loss"] = float(rows[-1].get("best_so_far_loss", "nan"))
                progress["progress_path"] = str(matches[0])
        except (OSError, ValueError):
            pass
        return progress


class ResultsDatabaseReader:
    """Read query-oriented views from an RL4CRN SQLite results database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.trial_id = self._find_trial_id()
        with self._connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.tables = tables
        with self._connect() as connection:
            self.table_columns = {
                table: {
                    str(row[1])
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                }
                for table in tables
            }
        required = {"training_runs", "crns", "evaluations", "hof_snapshots"}
        missing = sorted(required - tables)
        if missing:
            raise ValueError(f"Not an RL4CRN results database; missing tables: {missing}")

    def _find_trial_id(self) -> str:
        for parent in self.path.parents:
            manifest = parent / "trial_manifest.json"
            if not manifest.is_file():
                continue
            try:
                value = json.loads(manifest.read_text(encoding="utf-8")).get("trial_id")
            except (OSError, ValueError):
                continue
            if value:
                return str(value)
        return self.path.parent.name

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{quote(str(self.path))}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def runs(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT r.run_id, r.created_at, r.metadata_json,
                          COUNT(DISTINCT h.snapshot_id) AS snapshot_count,
                          MAX(h.epoch) AS latest_epoch
                   FROM training_runs r
                   LEFT JOIN hof_snapshots h ON h.run_id = r.run_id
                   GROUP BY r.run_id
                   ORDER BY r.created_at DESC"""
            ).fetchall()
        result = _rows(rows, json_fields={"metadata_json"})
        for run in result:
            metadata = run.get("metadata_json") or {}
            run["task"] = _task_label(metadata)
        return result

    def summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        run_clause = "" if run_id is None else " WHERE run_id = ?"
        params = () if run_id is None else (run_id,)
        with self._connect() as connection:
            run_count = connection.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            evaluation_count = connection.execute(
                f"SELECT COUNT(*) FROM evaluations{run_clause}", params
            ).fetchone()[0]
            optimization_count = connection.execute(
                f"SELECT COUNT(*) FROM optimization_runs{run_clause}", params
            ).fetchone()[0]
            llm_count = connection.execute(
                f"SELECT COUNT(*) FROM llm_candidates WHERE llm_run_id IN "
                f"(SELECT llm_run_id FROM llm_runs{run_clause})",
                params,
            ).fetchone()[0]
            llm_failure_count = 0
            if "llm_failures" in self.tables:
                llm_failure_count = connection.execute(
                    f"SELECT COUNT(*) FROM llm_failures{run_clause}", params
                ).fetchone()[0]
            snapshot_count = connection.execute(
                f"SELECT COUNT(*) FROM hof_snapshots{run_clause}", params
            ).fetchone()[0]
            latest_epoch = connection.execute(
                f"SELECT MAX(epoch) FROM hof_snapshots{run_clause}", params
            ).fetchone()[0]
            if run_id is None:
                best_loss = connection.execute(
                    """SELECT MIN(loss) FROM (
                           SELECT loss FROM evaluations WHERE valid = 1
                           UNION ALL
                           SELECT loss FROM hof_snapshot_entries
                       )"""
                ).fetchone()[0]
            else:
                best_loss = connection.execute(
                    """SELECT MIN(loss) FROM (
                           SELECT loss FROM evaluations WHERE valid = 1 AND run_id = ?
                           UNION ALL
                           SELECT e.loss FROM hof_snapshot_entries e
                           JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                           WHERE h.run_id = ?
                       )""",
                    (run_id, run_id),
                ).fetchone()[0]
            if run_id is None:
                topology_count = connection.execute("SELECT COUNT(*) FROM crns").fetchone()[0]
            else:
                topology_count = connection.execute(
                    """SELECT COUNT(*) FROM (
                           SELECT topology_hash FROM evaluations WHERE run_id = ?
                           UNION SELECT e.topology_hash FROM hof_snapshot_entries e
                               JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                               WHERE h.run_id = ?
                           UNION SELECT topology_hash FROM optimization_runs WHERE run_id = ?
                       )""",
                    (run_id, run_id, run_id),
                ).fetchone()[0]
        return {
            "database": str(self.path),
            "trial_id": self.trial_id,
            "run_count": run_count,
            "topology_count": topology_count,
            "evaluation_count": evaluation_count,
            "optimization_count": optimization_count,
            "llm_candidate_count": llm_count,
            "llm_failure_count": llm_failure_count,
            "snapshot_count": snapshot_count,
            "latest_epoch": latest_epoch,
            "best_loss": best_loss,
        }

    def loss_history(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        where = "" if run_id is None else "WHERE h.run_id = ?"
        params = () if run_id is None else (run_id,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT h.epoch, h.run_id, MIN(e.loss) AS best_loss,
                            AVG(e.loss) AS average_loss, COUNT(*) AS hof_size,
                            (SELECT best.topology_hash
                               FROM hof_snapshot_entries best
                              WHERE best.snapshot_id = h.snapshot_id
                              ORDER BY best.loss ASC, best.rank ASC LIMIT 1
                            ) AS best_topology_hash
                            ,(SELECT best.emitter
                                FROM hof_snapshot_entries best
                               WHERE best.snapshot_id = h.snapshot_id
                               ORDER BY best.loss ASC, best.rank ASC LIMIT 1
                             ) AS best_emitter
                     FROM hof_snapshots h
                     JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                     {where}
                     GROUP BY h.snapshot_id
                     ORDER BY h.epoch, h.created_at""",
                params,
            ).fetchall()
            sources = _initial_sources(
                connection,
                [str(row["best_topology_hash"]) for row in rows if row["best_topology_hash"]],
            )
        history = _rows(rows)
        for point in history:
            point["dominant_source"] = point.get("best_emitter") or sources.get(
                point.get("best_topology_hash"), "RL"
            )
        return history

    def latest_hof(self, run_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        where = "" if run_id is None else "WHERE run_id = ?"
        params = () if run_id is None else (run_id,)
        with self._connect() as connection:
            snapshot = connection.execute(
                f"""SELECT snapshot_id, run_id, epoch, created_at
                     FROM hof_snapshots {where}
                     ORDER BY epoch DESC, created_at DESC LIMIT 1""",
                params,
            ).fetchone()
            if snapshot is None:
                return {"snapshot": None, "entries": []}
            entries = connection.execute(
                """SELECT e.rank, e.topology_hash, e.loss, e.parameters_json,
                          e.task_info_json, e.emitter, e.provenance_class,
                          e.related_llm_proposal_id, e.related_llm_first_seen_epoch,
                          c.reaction_ids_json, c.crn_text
                   FROM hof_snapshot_entries e
                   JOIN crns c ON c.topology_hash = e.topology_hash
                   WHERE e.snapshot_id = ? ORDER BY e.rank LIMIT ?""",
                (snapshot["snapshot_id"], _limit(limit)),
            ).fetchall()
            sources = _initial_sources(
                connection, [str(entry["topology_hash"]) for entry in entries]
            )
        parsed_entries = _rows(
            entries,
            json_fields={"parameters_json", "task_info_json", "reaction_ids_json"},
        )
        for entry in parsed_entries:
            entry["initial_source"] = sources.get(entry["topology_hash"], "RL")
            entry["has_hof_plot"] = self.hof_plot(entry["topology_hash"]) is not None
        return {
            "snapshot": dict(snapshot),
            "entries": parsed_entries,
        }

    def optimizations(self, run_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        where = "" if run_id is None else "WHERE o.run_id = ?"
        params: tuple[Any, ...] = () if run_id is None else (run_id,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT o.optimization_id, o.run_id, o.epoch, o.hof_rank,
                            o.topology_hash, c.reaction_ids_json, o.original_loss,
                            o.optimized_loss, (o.original_loss - o.optimized_loss) AS improvement,
                            o.success, o.stored, o.elapsed_seconds, o.n_evaluations,
                            o.message, o.optimized_parameters_json
                     FROM optimization_runs o
                     JOIN crns c ON c.topology_hash = o.topology_hash
                     {where}
                     ORDER BY o.created_at DESC LIMIT ?""",
                params + (_limit(limit),),
            ).fetchall()
        return _rows(rows, json_fields={"reaction_ids_json", "optimized_parameters_json"})

    def llm_runs(self, run_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        where = "" if run_id is None else "WHERE run_id = ?"
        params: tuple[Any, ...] = () if run_id is None else (run_id,)
        with self._connect() as connection:
            if "response_validation_json" in self.table_columns.get("llm_runs", set()):
                completed_rows = connection.execute(
                    f"""SELECT llm_run_id, run_id, launched_epoch, completed_epoch,
                                requested, produced, valid_count, elapsed_seconds, created_at,
                                (SELECT MIN(c.loss) FROM llm_candidates c
                                  WHERE c.llm_run_id = llm_runs.llm_run_id
                                    AND c.valid = 1) AS best_loss,
                                returned, accepted, rejected, clamped_parameters,
                                provider_call_count, response_validation_json
                         FROM llm_runs {where}
                         ORDER BY created_at DESC LIMIT ?""",
                    params + (_limit(limit),),
                ).fetchall()
            else:
                completed_rows = connection.execute(
                    f"""SELECT llm_run_id, run_id, launched_epoch, completed_epoch,
                                requested, produced, valid_count, elapsed_seconds, created_at,
                                (SELECT MIN(c.loss) FROM llm_candidates c
                                  WHERE c.llm_run_id = llm_runs.llm_run_id
                                    AND c.valid = 1) AS best_loss
                         FROM llm_runs {where}
                         ORDER BY created_at DESC LIMIT ?""",
                    params + (_limit(limit),),
                ).fetchall()
            failed_rows = []
            if "llm_failures" in self.tables:
                failed_rows = connection.execute(
                    f"""SELECT failure_id AS llm_run_id, run_id, launched_epoch,
                                completed_epoch, requested,
                                COALESCE(returned, 0) AS produced,
                                COALESCE(accepted, 0) AS valid_count,
                                elapsed_seconds, created_at, NULL AS best_loss,
                                error, returned, accepted, rejected,
                                clamped_parameters, response_validation_json
                         FROM llm_failures {where}
                         ORDER BY created_at DESC LIMIT ?""",
                    params + (_limit(limit),),
                ).fetchall()
        runs = _rows(completed_rows, json_fields={"response_validation_json"})
        for run in runs:
            run["status"] = "completed"
            run["error"] = None
            run.setdefault("returned", run["produced"])
            run.setdefault("accepted", run["valid_count"])
            run.setdefault(
                "rejected",
                max(0, int(run["produced"]) - int(run["valid_count"])),
            )
            run.setdefault("clamped_parameters", None)
            run.setdefault("provider_call_count", None)
            run.setdefault("response_validation_json", {})
        failures = _rows(failed_rows, json_fields={"response_validation_json"})
        for failure in failures:
            failure["status"] = "failed"
        runs.extend(failures)
        runs.sort(key=lambda row: float(row.get("created_at") or 0.0), reverse=True)
        runs = runs[: _limit(limit)]
        _attach_workspace_reasoning(self.path, runs)
        return runs

    def llm_candidates(self, llm_run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT lc.candidate_index, lc.topology_hash, lc.candidate_json,
                          lc.valid, lc.loss, lc.message, lc.task_info_json,
                          c.reaction_ids_json, c.crn_text
                   FROM llm_candidates lc
                   LEFT JOIN crns c ON c.topology_hash = lc.topology_hash
                   WHERE lc.llm_run_id = ? ORDER BY lc.candidate_index""",
                (llm_run_id,),
            ).fetchall()
        candidates = _rows(
            rows,
            json_fields={"candidate_json", "task_info_json", "reaction_ids_json"},
        )
        for candidate in candidates:
            candidate["presentation"] = _parse_crn_presentation(
                candidate.get("crn_text", ""), candidate.get("reaction_ids_json") or []
            )
        return candidates

    def crns(self, run_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if run_id is None:
            selected_sql = "SELECT topology_hash FROM crns"
            params: tuple[Any, ...] = ()
            evaluation_filter = ""
        else:
            selected_sql = """SELECT topology_hash FROM evaluations WHERE run_id = ?
                UNION SELECT e.topology_hash FROM hof_snapshot_entries e
                    JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id WHERE h.run_id = ?
                UNION SELECT topology_hash FROM optimization_runs WHERE run_id = ?"""
            params = (run_id, run_id, run_id)
            evaluation_filter = "AND e.run_id = ?"
            params += (run_id,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""WITH selected AS ({selected_sql})
                     SELECT c.topology_hash, c.reaction_ids_json, c.crn_text,
                            MIN(e.loss) AS best_loss, COUNT(e.evaluation_id) AS evaluation_count,
                            GROUP_CONCAT(DISTINCT e.source) AS sources
                     FROM selected s
                     JOIN crns c ON c.topology_hash = s.topology_hash
                     LEFT JOIN evaluations e ON e.topology_hash = c.topology_hash {evaluation_filter}
                     GROUP BY c.topology_hash
                     ORDER BY best_loss IS NULL, best_loss ASC
                     LIMIT ?""",
                params + (_limit(limit),),
            ).fetchall()
            sources = _initial_sources(
                connection, [str(row["topology_hash"]) for row in rows]
            )
        parsed_rows = _rows(rows, json_fields={"reaction_ids_json"})
        for row in parsed_rows:
            row["initial_source"] = sources.get(row["topology_hash"], "RL")
        return parsed_rows

    def crn_detail(self, topology_hash: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            crn = connection.execute(
                "SELECT * FROM crns WHERE topology_hash = ?", (topology_hash,)
            ).fetchone()
            if crn is None:
                return None
            evaluations = connection.execute(
                """SELECT evaluation_id, run_id, source, epoch, loss, valid, message,
                          parameters_json, task_info_json, metadata_json, created_at
                   FROM evaluations WHERE topology_hash = ? ORDER BY loss IS NULL, loss""",
                (topology_hash,),
            ).fetchall()
            optimizations = connection.execute(
                """SELECT optimization_id, run_id, epoch, hof_rank, original_loss,
                          optimized_loss, success, stored, elapsed_seconds,
                          n_evaluations, message, optimized_parameters_json
                   FROM optimization_runs WHERE topology_hash = ? ORDER BY created_at DESC""",
                (topology_hash,),
            ).fetchall()
            initial_source = _initial_sources(connection, [topology_hash]).get(
                topology_hash, "RL"
            )
        crn_data = _row(crn, {"reaction_ids_json", "structure_json"})
        crn_data["initial_source"] = initial_source
        crn_data["has_hof_plot"] = self.hof_plot(topology_hash) is not None
        crn_data["presentation"] = _parse_crn_presentation(
            crn_data.get("crn_text", ""), crn_data.get("reaction_ids_json", [])
        )
        crn_data["evaluations"] = _rows(
            evaluations,
            json_fields={"parameters_json", "task_info_json", "metadata_json"},
        )
        crn_data["optimizations"] = _rows(
            optimizations, json_fields={"optimized_parameters_json"}
        )
        return crn_data

    def hof_plot(self, topology_hash: str) -> Optional[Path]:
        """Find the newest saved HoF trajectory plot for one topology."""

        exported = self.path.parent / "hof-plots" / f"{topology_hash}.jpg"
        if exported.is_file():
            return exported
        with self._connect() as connection:
            row = connection.execute(
                "SELECT crn_text FROM crns WHERE topology_hash = ?", (topology_hash,)
            ).fetchone()
        if row is None:
            return None
        workspace_root = self.path.parent / "harness-workspaces"
        for workspace in sorted(workspace_root.glob("*"), reverse=True):
            context_path = workspace / "CONTEXT/HALL_OF_FAME.json"
            try:
                entries = json.loads(context_path.read_text(encoding="utf-8")).get("entries", [])
            except (OSError, ValueError, AttributeError):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("topology_hash") != topology_hash and entry.get("crn") != row["crn_text"]:
                    continue
                rank = int(entry.get("rank", -1))
                artifact_root = workspace / "CONTEXT/hall-of-fame" / f"rank-{rank:03d}"
                for name in ("transients.jpg", "transients.jpeg", "transients.png"):
                    candidate = artifact_root / name
                    if candidate.is_file():
                        return candidate
        return None


def _initial_sources(
    connection: sqlite3.Connection, topology_hashes: List[str]
) -> Dict[str, str]:
    """Return the earliest known generator for each topology without modifying the DB."""

    hashes = list(dict.fromkeys(str(value) for value in topology_hashes if value))
    if not hashes:
        return {}
    placeholders = ",".join("?" for _ in hashes)
    events: Dict[str, List[tuple[float, str]]] = {value: [] for value in hashes}
    evaluation_rows = connection.execute(
        f"""SELECT topology_hash, source, created_at
              FROM evaluations WHERE topology_hash IN ({placeholders})""",
        hashes,
    ).fetchall()
    for row in evaluation_rows:
        events[str(row["topology_hash"])].append(
            (float(row["created_at"]), _source_label(row["source"], fallback="RL"))
        )
    snapshot_rows = connection.execute(
        f"""SELECT e.topology_hash, e.task_info_json, h.created_at
              FROM hof_snapshot_entries e
              JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
             WHERE e.topology_hash IN ({placeholders})""",
        hashes,
    ).fetchall()
    for row in snapshot_rows:
        try:
            task_info = json.loads(row["task_info_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            task_info = {}
        events[str(row["topology_hash"])].append(
            (
                float(row["created_at"]),
                _source_label(task_info.get("source"), fallback="RL"),
            )
        )
    return {
        topology_hash: min(source_events, key=lambda item: item[0])[1]
        if source_events
        else "RL"
        for topology_hash, source_events in events.items()
    }


def _source_label(value: Any, *, fallback: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized == "LLM":
        return "LLM"
    if normalized == "RL":
        return "RL"
    return fallback


def _attach_workspace_reasoning(database_path: Path, runs: List[Dict[str, Any]]) -> None:
    """Match completed DB rounds to nearby completed Harness workspaces, read-only."""

    workspace_root = database_path.parent / "harness-workspaces"
    if not workspace_root.is_dir():
        for run in runs:
            run["reasoning_requests"] = []
            run["reasoning_workspace"] = None
        return
    workspaces = []
    for workspace in sorted(workspace_root.iterdir()):
        status_path = workspace / "run_status.json"
        notes_path = workspace / "REASONING_NOTES.md"
        summary_path = workspace / "evaluation_summary.json"
        if not (status_path.is_file() and notes_path.is_file() and summary_path.is_file()):
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            notes = notes_path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") != "completed":
            continue
        workspaces.append(
            {
                "path": str(workspace),
                "completed_at": status_path.stat().st_mtime,
                "candidate_count": int(summary.get("candidate_count", -1)),
                "reasoning_requests": _reasoning_sections(notes),
            }
        )
    unused = list(workspaces)
    for run in sorted(runs, key=lambda item: float(item.get("created_at") or 0)):
        produced = int(run.get("produced") or 0)
        matching = [item for item in unused if item["candidate_count"] == produced] or unused
        if not matching:
            run["reasoning_requests"] = []
            run["reasoning_workspace"] = None
            continue
        workspace = min(
            matching,
            key=lambda item: abs(item["completed_at"] - float(run.get("created_at") or 0)),
        )
        unused.remove(workspace)
        run["reasoning_requests"] = workspace["reasoning_requests"]
        run["reasoning_workspace"] = workspace["path"]


def _reasoning_sections(notes: str) -> List[Dict[str, str]]:
    """Extract concise request-level sections from a reasoning-notes artifact."""

    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", str(notes)))
    sections = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        if not title.lower().startswith("call"):
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(notes)
        content = notes[match.end() : end].strip()
        if content:
            sections.append({"title": title, "content": content})
    if sections:
        return sections
    cleaned = str(notes).strip()
    if not cleaned:
        return []
    return [{"title": "Proposal reasoning", "content": cleaned}]


def _parse_crn_presentation(crn_text: str, reaction_ids: List[int]) -> Dict[str, Any]:
    """Convert the stable CRN text representation into display-ready equations."""

    metadata: Dict[str, List[str]] = {"inputs": [], "species": [], "outputs": []}
    reactions: List[Dict[str, Any]] = []
    header_names = {
        "Inputs": "inputs",
        "Species": "species",
        "Output Species": "outputs",
    }
    reaction_index = 0
    for raw_line in str(crn_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched_header = False
        for prefix, key in header_names.items():
            marker = prefix + ":"
            if line.startswith(marker):
                try:
                    values = ast.literal_eval(line[len(marker) :].strip())
                    metadata[key] = [str(value) for value in values]
                except (SyntaxError, ValueError, TypeError):
                    metadata[key] = []
                matched_header = True
                break
        if matched_header or "---->" not in line:
            continue
        equation, _, kinetics = line.partition(";")
        reactant_text, product_text = equation.split("---->", 1)
        rate, inputs = _parse_mass_action(kinetics)
        reactants = _species_terms(reactant_text)
        products = _species_terms(product_text)
        reaction_id = reaction_ids[reaction_index] if reaction_index < len(reaction_ids) else None
        reaction = {
            "reaction_id": reaction_id,
            "reactants": reactants,
            "products": products,
            "rate": rate,
            "inputs": inputs,
            "kind": "template" if inputs else "designed",
        }
        reaction["latex"] = _reaction_latex(reaction)
        reactions.append(reaction)
        reaction_index += 1
    return {
        **metadata,
        "reactions": reactions,
        "latex": "\\begin{aligned}\n"
        + " \\\\\n".join(reaction["latex"] for reaction in reactions)
        + "\n\\end{aligned}",
    }


def _parse_mass_action(kinetics: str) -> tuple[Optional[float], List[str]]:
    match = re.search(r"MAK\((.*)\)", kinetics)
    if match is None:
        return None, []
    fields = [field.strip() for field in match.group(1).split(",")]
    try:
        rate = float(fields[0])
    except (IndexError, ValueError):
        rate = None
    return rate, [field for field in fields[1:] if field and field != "None"]


def _species_terms(text: str) -> List[Dict[str, Any]]:
    names = [name.strip() for name in text.strip().split("+") if name.strip()]
    if not names or names == ["∅"]:
        return []
    ordered: List[str] = []
    counts: Dict[str, int] = {}
    for name in names:
        if name not in counts:
            ordered.append(name)
            counts[name] = 0
        counts[name] += 1
    return [{"species": name, "coefficient": counts[name]} for name in ordered]


def _reaction_latex(reaction: Dict[str, Any]) -> str:
    def side(terms: List[Dict[str, Any]]) -> str:
        if not terms:
            return r"\varnothing"
        return " + ".join(
            ((str(term["coefficient"]) + r"\,") if term["coefficient"] > 1 else "")
            + _species_latex(term["species"])
            for term in terms
        )

    annotations = []
    if reaction.get("rate") is not None:
        annotations.append(f"k={reaction['rate']:.6g}")
    annotations.extend(_species_latex(value) for value in reaction.get("inputs", []))
    label = ",\\;".join(annotations)
    return f"{side(reaction['reactants'])} &\\xrightarrow{{{label}}} {side(reaction['products'])}"


def _species_latex(name: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)_([A-Za-z0-9]+)", str(name))
    if match:
        return rf"\mathrm{{{match.group(1)}}}_{{{match.group(2)}}}"
    safe = re.sub(r"[^A-Za-z0-9 ]", "", str(name)) or "?"
    return rf"\mathrm{{{safe}}}"


def _limit(value: Any) -> int:
    try:
        return max(1, min(int(value), 1000))
    except (TypeError, ValueError):
        return 200


def _task_label(metadata: Dict[str, Any]) -> str:
    for key in ("task", "task_name", "task_kind", "notebook"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "Unlabeled"


def _row(row: sqlite3.Row, json_fields: set[str] | None = None) -> Dict[str, Any]:
    result = dict(row)
    for field in json_fields or set():
        if result.get(field) is not None:
            try:
                result[field] = json.loads(result[field])
            except (TypeError, json.JSONDecodeError):
                pass
    return result


def _rows(rows: Any, json_fields: set[str] | None = None) -> List[Dict[str, Any]]:
    return [_row(row, json_fields) for row in rows]


def _handler(
    reader: ResultsDatabaseReader,
    access_token: Optional[str] = None,
    campaign_reader: Optional[CampaignProgressReader] = None,
) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if access_token is not None and _query_value(query, "token") != access_token:
                self._json(403, {"error": "invalid or missing access token"})
                return
            if parsed.path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", _HTML.encode("utf-8"))
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if parsed.path == "/api/hof-plot":
                try:
                    plot = reader.hof_plot(_required_query(query, "topology_hash"))
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                if plot is None:
                    self._json(404, {"error": "HoF plot not found"})
                    return
                content_type = "image/png" if plot.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, content_type, plot.read_bytes())
                return
            if not parsed.path.startswith("/api/"):
                self._json(404, {"error": "not found"})
                return
            run_id = _query_value(query, "run_id")
            limit = _limit(_query_value(query, "limit") or 200)
            try:
                routes = {
                    "/api/campaigns": lambda: (
                        campaign_reader.snapshot() if campaign_reader is not None else None
                    ),
                    "/api/runs": lambda: reader.runs(),
                    "/api/summary": lambda: reader.summary(run_id),
                    "/api/history": lambda: reader.loss_history(run_id),
                    "/api/hof": lambda: reader.latest_hof(run_id, limit),
                    "/api/optimizations": lambda: reader.optimizations(run_id, limit),
                    "/api/llm": lambda: reader.llm_runs(run_id, limit),
                    "/api/llm-candidates": lambda: reader.llm_candidates(
                        _required_query(query, "llm_run_id")
                    ),
                    "/api/crns": lambda: reader.crns(run_id, limit),
                    "/api/crn": lambda: reader.crn_detail(
                        _required_query(query, "topology_hash")
                    ),
                }
                route = routes.get(parsed.path)
                if route is None:
                    self._json(404, {"error": "unknown API route"})
                    return
                data = route()
                if data is None:
                    self._json(404, {"error": "record not found"})
                    return
                self._json(200, data)
            except (ValueError, sqlite3.Error) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, status: int, value: Any) -> None:
            self._send(
                status,
                "application/json; charset=utf-8",
                json.dumps(value, allow_nan=False).encode("utf-8"),
            )

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return ViewerHandler


def _query_value(query: Dict[str, List[str]], name: str) -> Optional[str]:
    values = query.get(name, [])
    value = values[0].strip() if values else ""
    return value or None


def _required_query(query: Dict[str, List[str]], name: str) -> str:
    value = _query_value(query, name)
    if value is None:
        raise ValueError(f"Missing query parameter: {name}")
    return value


def serve_results_database(
    path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    access_token: Optional[str] = None,
    campaign_plan: Optional[str | Path] = None,
) -> None:
    """Serve the viewer until interrupted."""

    reader = ResultsDatabaseReader(path)
    campaign_reader = CampaignProgressReader(campaign_plan) if campaign_plan else None
    if access_token == "auto":
        access_token = secrets.token_urlsafe(24)
    if host not in {"127.0.0.1", "localhost", "::1"} and not access_token:
        raise ValueError("A non-loopback viewer requires --token auto or an explicit token.")
    server = ThreadingHTTPServer(
        (host, int(port)), _handler(reader, access_token, campaign_reader)
    )
    url = f"http://{host}:{server.server_port}"
    if access_token:
        url += f"/?token={quote(access_token)}"
    print(f"RL4CRN results viewer: {url}")
    print(f"Database: {reader.path}")
    if campaign_reader is not None:
        print(f"Campaign plan: {campaign_reader.path}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping results viewer.")
    finally:
        server.server_close()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="View an RL4CRN results SQLite database.")
    parser.add_argument("database", help="Path to the results SQLite database")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument(
        "--token",
        help="Access token for remote viewers; use 'auto' to generate one",
    )
    parser.add_argument("--campaign-plan", help="Optional paper campaign plan JSON")
    args = parser.parse_args(argv)
    serve_results_database(
        args.database,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        access_token=args.token,
        campaign_plan=args.campaign_plan,
    )


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL4CRN Results</title>
  <style>
    :root { color-scheme: light; --ink:#17211c; --muted:#64706a; --line:#d9dfdc; --paper:#ffffff; --wash:#f3f6f4; --green:#16794b; --blue:#2463a5; --red:#b6403a; --amber:#9a6500; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--wash); color:var(--ink); font:14px/1.45 system-ui, sans-serif; letter-spacing:0; }
    header { height:64px; display:flex; align-items:center; gap:18px; padding:0 24px; background:var(--paper); border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:20px; font-weight:700; letter-spacing:0; }
    .db { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
    button, select { min-height:34px; border:1px solid #b8c2bd; background:var(--paper); color:var(--ink); border-radius:6px; padding:6px 10px; font:inherit; letter-spacing:0; }
    button { cursor:pointer; }
    button:hover, button.active { border-color:var(--green); color:var(--green); }
    .layout { display:grid; grid-template-columns:180px minmax(0,1fr); min-height:calc(100vh - 64px); }
    nav { padding:18px 12px; border-right:1px solid var(--line); background:#f9fbfa; }
    nav button { width:100%; text-align:left; border-color:transparent; background:transparent; margin-bottom:4px; }
    nav button.active { background:#e8f2ed; border-color:#c6ddd1; }
    main { min-width:0; padding:22px 26px 40px; }
    .toolbar { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
    .toolbar label { color:var(--muted); }
    #runSelect { flex:1; min-width:0; }
    .metrics { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); border:1px solid var(--line); background:var(--paper); margin-bottom:18px; }
    .metric { padding:14px 16px; border-right:1px solid var(--line); }
    .metric:last-child { border-right:0; }
    .metric span { display:block; color:var(--muted); font-size:12px; }
    .metric strong { display:block; margin-top:3px; font-size:21px; letter-spacing:0; }
    section { display:none; }
    section.active { display:block; }
    .panel { background:var(--paper); border:1px solid var(--line); border-radius:6px; margin-bottom:18px; overflow:hidden; }
    .panel-head { display:flex; align-items:center; justify-content:space-between; min-height:46px; padding:10px 14px; border-bottom:1px solid var(--line); }
    h2 { font-size:15px; margin:0; letter-spacing:0; }
    .chart-wrap { height:260px; padding:14px; }
    canvas { width:100%; height:100%; display:block; }
    .table-wrap { overflow:auto; max-height:65vh; }
    table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
    th { position:sticky; top:0; z-index:1; background:#f4f7f5; color:#4d5953; font-size:12px; text-align:left; }
    th, td { padding:9px 12px; border-bottom:1px solid #e5e9e7; white-space:nowrap; }
    tbody tr { cursor:pointer; }
    tbody tr:hover { background:#f0f7f3; }
    .hash { font-family:ui-monospace, monospace; color:var(--blue); }
    .good { color:var(--green); font-weight:650; }
    .bad { color:var(--red); font-weight:650; }
    .source-llm { color:var(--blue); font-weight:700; }
    .source-rl { color:var(--green); font-weight:700; }
    .empty { padding:32px; text-align:center; color:var(--muted); }
    .campaign-summary { display:grid; grid-template-columns:repeat(6,minmax(105px,1fr)); border:1px solid var(--line); background:var(--paper); margin-bottom:18px; }
    .campaign-list { display:grid; gap:10px; }
    .campaign-row { display:grid; grid-template-columns:minmax(220px,1.4fr) minmax(180px,1fr) 160px 130px; gap:18px; align-items:center; padding:14px 16px; border-bottom:1px solid #e5e9e7; }
    .campaign-row:last-child { border-bottom:0; }
    .campaign-name strong, .campaign-counts strong { display:block; font-size:14px; }
    .campaign-name span, .campaign-counts span { color:var(--muted); font-size:12px; }
    .progress-track { height:9px; background:#e5e9e7; overflow:hidden; border-radius:4px; }
    .progress-fill { display:block; height:100%; background:var(--green); transition:width .25s ease; }
    .progress-label { display:flex; justify-content:space-between; margin-bottom:5px; color:var(--muted); font-size:12px; }
    .phase { justify-self:start; border:1px solid var(--line); border-radius:5px; padding:4px 7px; font-size:12px; font-weight:700; text-transform:capitalize; }
    .phase.running { color:var(--blue); border-color:#a9c4df; background:#f1f7fc; }
    .phase.completed { color:var(--green); border-color:#acd0bd; background:#eff8f3; }
    .phase.attention { color:var(--red); border-color:#e5b7b4; background:#fff3f2; }
    .llm-batches { background:var(--paper); }
    .llm-batch { width:100%; display:grid; grid-template-columns:130px minmax(210px,1fr) minmax(180px,1.2fr) 150px 28px; align-items:center; gap:16px; min-height:88px; padding:12px 16px; border:0; border-bottom:1px solid #e5e9e7; border-radius:0; text-align:left; }
    .llm-batch:last-child { border-bottom:0; }
    .llm-batch:hover { background:#f0f7f3; color:var(--ink); }
    .llm-batch.failed { background:#fff7f6; border-left:3px solid var(--red); }
    .epoch-span strong, .batch-title strong { display:block; font-size:15px; }
    .epoch-span span, .batch-title span, .batch-stat span { display:block; color:var(--muted); font-size:12px; }
    .batch-title code { color:var(--blue); font-size:12px; }
    .validity-line { display:flex; justify-content:space-between; gap:10px; margin-bottom:6px; font-size:12px; }
    .validity-track { height:7px; background:#e5e9e7; border-radius:4px; overflow:hidden; }
    .validity-fill { display:block; height:100%; background:var(--green); }
    .batch-stat strong { display:block; font-size:14px; font-variant-numeric:tabular-nums; }
    .batch-open { color:var(--blue); font-size:20px; text-align:right; }
    .candidate-list { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .candidate-row { display:grid; grid-template-columns:48px minmax(260px,1fr) 120px 112px; gap:12px; align-items:center; min-height:72px; padding:9px 12px; border-bottom:1px solid #e5e9e7; }
    .candidate-row:last-child { border-bottom:0; }
    .candidate-row.invalid { background:#fff7f6; }
    .candidate-index { color:var(--muted); font:12px ui-monospace,monospace; }
    .candidate-main strong { display:block; font:13px ui-monospace,monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .candidate-main span { display:block; margin-top:4px; color:var(--muted); font:11px ui-monospace,monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .status-pill { display:inline-block; border-radius:5px; padding:3px 7px; font-size:12px; font-weight:700; }
    .status-pill.valid { background:#e7f3ec; color:var(--green); }
    .status-pill.invalid { background:#fff0ef; color:var(--red); }
    .reasoning-section { margin:4px 0 20px; }
    .reasoning-section > h3 { margin:0 0 8px; font-size:13px; color:#4d5953; }
    .reasoning-request { padding:12px 14px; border-left:3px solid var(--blue); border-bottom:1px solid #e5e9e7; background:#f8fafc; }
    .reasoning-request:last-of-type { border-bottom:0; }
    .reasoning-request h4 { margin:0 0 8px; font-size:14px; letter-spacing:0; }
    .reasoning-request p { margin:6px 0; color:#34413a; }
    .reasoning-request ul { margin:6px 0; padding-left:20px; }
    .reasoning-request code { font:12px ui-monospace,monospace; color:var(--blue); }
    .reasoning-path { display:block; margin-top:8px; color:var(--muted); font:11px ui-monospace,monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    dialog { width:min(1080px,calc(100vw - 32px)); max-height:88vh; border:1px solid #aeb8b3; border-radius:7px; padding:0; color:var(--ink); }
    dialog::backdrop { background:rgba(20,30,25,.35); }
    .dialog-head { display:flex; align-items:center; justify-content:space-between; padding:12px 16px; border-bottom:1px solid var(--line); }
    pre { margin:0; padding:16px; overflow:auto; font:12px/1.5 ui-monospace, monospace; white-space:pre-wrap; word-break:break-word; }
    .detail-body { padding:18px; overflow:auto; max-height:calc(88vh - 54px); }
    .detail-body > pre { margin:-18px; }
    .crn-meta { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px; }
    .meta-chip { border:1px solid var(--line); background:#f5f8f6; border-radius:5px; padding:5px 8px; color:#4d5953; }
    .reaction-section { margin-top:18px; }
    .reaction-section h3 { margin:0 0 8px; font-size:13px; color:#4d5953; letter-spacing:0; }
    .reaction-list { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .reaction { display:grid; grid-template-columns:64px minmax(280px,1fr) 120px; align-items:center; min-height:58px; border-bottom:1px solid #e5e9e7; background:#fff; }
    .reaction:last-child { border-bottom:0; }
    .reaction.template { background:#f7fafc; }
    .reaction-id { padding:0 10px; color:var(--muted); font:12px ui-monospace,monospace; }
    .equation { display:flex; align-items:center; justify-content:center; gap:12px; padding:8px; font:17px/1.25 Georgia,'Times New Roman',serif; min-width:0; }
    .reaction-side { flex:1; display:flex; justify-content:flex-end; align-items:baseline; gap:5px; flex-wrap:wrap; }
    .reaction-side.products { justify-content:flex-start; }
    .species.output { color:var(--green); font-weight:700; }
    .coefficient { color:#323d37; }
    .plus { color:#7b8680; }
    .reaction-arrow { width:106px; flex:0 0 106px; text-align:center; }
    .arrow-label { display:block; min-height:18px; color:var(--blue); font:11px/1.2 ui-monospace,monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .arrow-line { display:block; position:relative; border-top:1.5px solid #47534d; margin-top:4px; }
    .arrow-line::after { content:''; position:absolute; right:-1px; top:-4px; border-left:7px solid #47534d; border-top:3px solid transparent; border-bottom:3px solid transparent; }
    .reaction-kind { padding:0 12px; color:var(--muted); font-size:12px; text-align:right; }
    .latex-box { margin-top:18px; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .trajectory-figure { margin:18px 0 0; border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#fff; }
    .trajectory-figure figcaption { padding:8px 12px; background:#f4f7f5; color:#4d5953; font-size:12px; font-weight:700; }
    .trajectory-figure img { display:block; width:100%; height:auto; max-height:520px; object-fit:contain; }
    .latex-head { display:flex; justify-content:space-between; align-items:center; min-height:40px; padding:5px 10px; background:#f4f7f5; }
    .latex-box pre { max-height:180px; padding:10px; background:#fbfcfb; }
    details.raw-record { margin-top:16px; border-top:1px solid var(--line); padding-top:12px; }
    details.raw-record summary { cursor:pointer; color:var(--blue); }
    .error { padding:12px 14px; border:1px solid #e5b7b4; background:#fff3f2; color:#8d2823; margin-bottom:14px; }
    @media (max-width:900px) { .layout { grid-template-columns:1fr; } nav { display:flex; overflow:auto; border-right:0; border-bottom:1px solid var(--line); padding:8px; } nav button { width:auto; margin:0 4px 0 0; white-space:nowrap; } main { padding:16px; } .metrics,.campaign-summary { grid-template-columns:repeat(2,1fr); } .metric:nth-child(2n) { border-right:0; } header { padding:0 14px; } .campaign-row { grid-template-columns:1fr; gap:8px; } .reaction { grid-template-columns:48px minmax(0,1fr); } .reaction-kind { display:none; } .equation { font-size:14px; gap:7px; } .reaction-arrow { width:72px; flex-basis:72px; } .llm-batch { grid-template-columns:100px minmax(160px,1fr) 120px 22px; } .llm-batch .batch-validity { display:none; } .candidate-row { grid-template-columns:36px minmax(150px,1fr) 90px; } .candidate-row button { display:none; } }
  </style>
</head>
<body>
  <header><h1>RL4CRN Results</h1><div class="db" id="dbPath">Loading database...</div><button id="refresh">Refresh</button></header>
  <div class="layout">
    <nav id="tabs">
      <button data-view="campaigns">Experiments</button>
      <button class="active" data-view="overview">Overview</button>
      <button data-view="hof">Hall of Fame</button>
      <button data-view="optimizations">Optimizations</button>
      <button data-view="llm">LLM runs</button>
      <button data-view="crns">Saved CRNs</button>
    </nav>
    <main>
      <div class="toolbar"><label for="taskSelect">Task</label><select id="taskSelect"><option value="">All tasks</option></select><label for="runSelect">Run</label><select id="runSelect"><option value="">All runs</option></select><span id="status"></span></div>
      <div id="error"></div>
      <section id="campaigns">
        <div class="campaign-summary" id="campaignSummary"></div>
        <div class="panel"><div class="panel-head"><h2>Paper experiment queue</h2><span id="campaignUpdated"></span></div><div class="campaign-list" id="campaignList"></div></div>
      </section>
      <section id="overview" class="active">
        <div class="metrics" id="metrics"></div>
        <div class="panel"><div class="panel-head"><h2>Best HoF loss by epoch</h2></div><div class="chart-wrap"><canvas id="lossChart"></canvas></div></div>
      </section>
      <section id="hof"><div class="panel"><div class="panel-head"><h2>Latest Hall of Fame</h2><span id="hofMeta"></span></div><div class="table-wrap" id="hofTable"></div></div></section>
      <section id="optimizations"><div class="panel"><div class="panel-head"><h2>Parameter optimization runs</h2></div><div class="table-wrap" id="optimizationTable"></div></div></section>
      <section id="llm"><div class="panel"><div class="panel-head"><h2>LLM proposal batches</h2><span id="llmMeta"></span></div><div class="llm-batches" id="llmRuns"></div></div></section>
      <section id="crns"><div class="panel"><div class="panel-head"><h2>Saved CRN topologies</h2></div><div class="table-wrap" id="crnTable"></div></div></section>
    </main>
  </div>
  <dialog id="detail"><div class="dialog-head"><h2 id="detailTitle">Details</h2><button id="closeDetail" aria-label="Close">Close</button></div><div id="detailBody" class="detail-body"></div></dialog>
  <script>
    const pageParams = new URLSearchParams(location.search);
    const requestedView = ['campaigns','overview','hof','optimizations','llm','crns'].includes(pageParams.get('view')) ? pageParams.get('view') : 'overview';
    const state = { view:requestedView, task:'', run:'', runs:[], history:[] };
    const $ = id => document.getElementById(id);
    const fmt = v => v === null || v === undefined ? '-' : typeof v === 'number' ? (Math.abs(v) < .001 && v !== 0 ? v.toExponential(3) : v.toLocaleString(undefined,{maximumSignificantDigits:6})) : String(v);
    const duration = seconds => { if(seconds===null||seconds===undefined)return '-'; if(seconds<60)return `${seconds.toFixed(1)} s`; if(seconds<3600){const minutes=Math.floor(seconds/60),rest=Math.round(seconds%60);return `${minutes}m ${rest}s`;}if(seconds<86400){const hours=Math.floor(seconds/3600),minutes=Math.round((seconds%3600)/60);return `${hours}h ${minutes}m`;}const days=Math.floor(seconds/86400),hours=Math.round((seconds%86400)/3600);return `${days}d ${hours}h`; };
    const shortHash = v => v ? v.slice(0,12) : '-';
    const accessToken = pageParams.get('token') || '';
    function assetUrl(path,params={}){const q=new URLSearchParams(params);if(accessToken)q.set('token',accessToken);return path+(q.size?'?'+q:'');}
    async function api(path, params={}) { const q = new URLSearchParams(params); if(accessToken)q.set('token',accessToken); const r = await fetch(path+(q.size?'?'+q:'')); const data = await r.json(); if(!r.ok) throw new Error(data.error||'Request failed'); return data; }
    function showError(err) { $('error').innerHTML = err ? `<div class="error"></div>` : ''; if(err) $('error').firstElementChild.textContent=err.message||err; }
    function table(target, columns, rows, onClick) { const root=$(target); if(!rows.length){root.innerHTML='<div class="empty">No records yet</div>'; return;} const t=document.createElement('table'); const h=t.createTHead().insertRow(); columns.forEach(c=>{const th=document.createElement('th');th.textContent=c[0];h.appendChild(th)}); const b=t.createTBody(); rows.forEach(row=>{const tr=b.insertRow(); columns.forEach(c=>{const td=tr.insertCell(); const value=c[1](row); td.textContent=value.text ?? value; if(value.className)td.className=value.className;}); if(onClick)tr.onclick=()=>onClick(row)}); root.replaceChildren(t); }
    const value = (text,className='') => ({text:fmt(text),className});
    async function loadRuns(){ state.runs=await api('/api/runs'); const tasks=[...new Set(state.runs.map(r=>r.task))].sort(); for(const task of tasks){const o=document.createElement('option');o.value=task;o.textContent=task;$('taskSelect').appendChild(o)} renderRuns(); }
    function renderRuns(){ const select=$('runSelect');select.replaceChildren();const matching=state.task?state.runs.filter(r=>r.task===state.task):state.runs;if(!state.task){const all=document.createElement('option');all.value='';all.textContent='All runs';select.appendChild(all)}for(const r of matching){const o=document.createElement('option');o.value=r.run_id;o.textContent=`${r.run_id} | epoch ${r.latest_epoch ?? '-'}`;select.appendChild(o)}if(state.task&&matching.length){state.run=matching[0].run_id;select.value=state.run}else{state.run='';select.value='';} }
    async function loadOverview(){ const [s,h]=await Promise.all([api('/api/summary',{run_id:state.run}),api('/api/history',{run_id:state.run})]); $('dbPath').textContent=`Trial ${s.trial_id} | ${s.database}`; const items=[['Best loss',fmt(s.best_loss)],['Topologies',s.topology_count],['Evaluations',s.evaluation_count],['Optimizations',s.optimization_count],['LLM candidates',s.llm_candidate_count],['Failed LLM rounds',s.llm_failure_count],['Latest epoch',s.latest_epoch ?? '-']]; $('metrics').innerHTML=items.map(x=>`<div class="metric"><span>${x[0]}</span><strong>${x[1]}</strong></div>`).join(''); state.history=h; drawChart(); }
    async function loadCampaigns(){const data=await api('/api/campaigns');const totals=data.totals;const summary=[['Plan progress',`${totals.progress_percent.toFixed(1)}%`],['Runs',totals.runs],['Completed',totals.completed],['Active',totals.active],['Pending',totals.pending],['Failed',totals.failed]];$('campaignSummary').innerHTML=summary.map(x=>`<div class="metric"><span>${x[0]}</span><strong>${x[1]}</strong></div>`).join('');$('campaignUpdated').textContent=`updated ${new Date(data.updated_at).toLocaleTimeString()}`;const root=$('campaignList');root.replaceChildren();data.campaigns.forEach(item=>{const row=document.createElement('div');row.className='campaign-row';const name=document.createElement('div');name.className='campaign-name';const title=document.createElement('strong');title.textContent=item.label;const details=document.createElement('span');details.textContent=`${item.model} | ${item.tasks.join(', ')} | ${item.seeds} seeds | ${item.epochs} epochs`;name.append(title,details);const progress=document.createElement('div');const label=document.createElement('div');label.className='progress-label';const pct=Math.max(0,Math.min(100,item.progress_percent));const runTotal=item.total_runs||item.potential_runs;label.innerHTML=`<span>${pct.toFixed(1)}%</span><span>${item.completed}/${runTotal} ${item.conditional&&!item.total_runs?'potential ':''}runs</span>`;const track=document.createElement('div');track.className='progress-track';const fill=document.createElement('span');fill.className='progress-fill';fill.style.width=`${pct}%`;track.appendChild(fill);progress.append(label,track);const counts=document.createElement('div');counts.className='campaign-counts';const count=document.createElement('strong');count.textContent=`${item.active} active · ${item.pending} queued`;const note=document.createElement('span');note.textContent=item.eta_seconds!==null?`ETA ${duration(item.eta_seconds)}`:(item.variant||'standard');counts.append(count,note);const phase=document.createElement('span');phase.className=`phase ${item.phase}`;phase.textContent=item.phase;row.append(name,progress,counts,phase);root.appendChild(row)});}
    function drawChart(){
      const canvas=$('lossChart'),dpr=devicePixelRatio||1,box=canvas.getBoundingClientRect();
      canvas.width=Math.max(1,box.width*dpr);canvas.height=Math.max(1,box.height*dpr);
      const c=canvas.getContext('2d');c.scale(dpr,dpr);
      const w=box.width,h=box.height,p={l:58,r:18,t:30,b:34};c.clearRect(0,0,w,h);
      const rows=state.history.filter(x=>Number.isFinite(x.best_loss));
      if(!rows.length){c.fillStyle='#64706a';c.fillText('No HoF snapshots yet',p.l,p.t+20);return;}
      const sourceColor=source=>source==='LLM'?'#2463a5':'#16794b';
      const sourceWash=source=>source==='LLM'?'rgba(36,99,165,.07)':'rgba(22,121,75,.06)';
      const xs=rows.map(x=>x.epoch),ys=rows.map(x=>x.best_loss),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),dx=xmax-xmin||1,dy=ymax-ymin||1;
      const X=x=>p.l+(x-xmin)/dx*(w-p.l-p.r),Y=y=>p.t+(ymax-y)/dy*(h-p.t-p.b);
      rows.forEach((row,index)=>{const left=index?((X(rows[index-1].epoch)+X(row.epoch))/2):p.l;const right=index<rows.length-1?((X(row.epoch)+X(rows[index+1].epoch))/2):(w-p.r);c.fillStyle=sourceWash(row.dominant_source);c.fillRect(left,p.t,Math.max(1,right-left),h-p.t-p.b);});
      c.strokeStyle='#d9dfdc';c.lineWidth=1;c.beginPath();c.moveTo(p.l,p.t);c.lineTo(p.l,h-p.b);c.lineTo(w-p.r,h-p.b);c.stroke();
      c.fillStyle='#64706a';c.font='12px system-ui, sans-serif';c.fillText(fmt(ymax),4,p.t+4);c.fillText(fmt(ymin),4,h-p.b+4);c.fillText(fmt(xmin),p.l,h-8);c.fillText(fmt(xmax),w-p.r-24,h-8);
      for(let index=1;index<rows.length;index++){const previous=rows[index-1],row=rows[index];c.strokeStyle=sourceColor(row.dominant_source);c.lineWidth=2.5;c.beginPath();c.moveTo(X(previous.epoch),Y(previous.best_loss));c.lineTo(X(row.epoch),Y(row.best_loss));c.stroke();}
      rows.forEach(row=>{c.fillStyle='#fff';c.strokeStyle=sourceColor(row.dominant_source);c.lineWidth=2;c.beginPath();c.arc(X(row.epoch),Y(row.best_loss),3,0,Math.PI*2);c.fill();c.stroke();});
      const legend=[['RL incumbent','#16794b'],['LLM incumbent','#2463a5']];let legendX=w-p.r-210;legend.forEach(([label,color],index)=>{const x=legendX+index*108;c.fillStyle=color;c.fillRect(x,8,12,3);c.fillStyle='#4d5953';c.fillText(label,x+18,12);});
    }
    async function loadHof(){ const d=await api('/api/hof',{run_id:state.run}); $('hofMeta').textContent=d.snapshot?`run ${d.snapshot.run_id} | epoch ${d.snapshot.epoch}`:''; table('hofTable',[['Rank',r=>value(r.rank+1)],['Loss',r=>value(r.loss,'good')],['Emitter',r=>value(r.emitter,`source-${String(r.emitter).toLowerCase()}`)],['Provenance',r=>value(String(r.provenance_class||'').replaceAll('_',' '))],['Reaction IDs',r=>value(JSON.stringify(r.reaction_ids_json))],['Topology',r=>value(shortHash(r.topology_hash),'hash')],['Network',r=>value(`${r.reaction_ids_json.length} reactions`)]],d.entries,r=>showCrn(r.topology_hash)); }
    async function loadOptimizations(){ const rows=await api('/api/optimizations',{run_id:state.run}); table('optimizationTable',[['Epoch',r=>value(r.epoch)],['Rank',r=>value(r.hof_rank+1)],['Reaction IDs',r=>value(JSON.stringify(r.reaction_ids_json))],['Before',r=>value(r.original_loss)],['After',r=>value(r.optimized_loss,'good')],['Improvement',r=>value(r.improvement,r.improvement>0?'good':'bad')],['Success',r=>value(r.success?'yes':'no',r.success?'good':'bad')],['Seconds',r=>value(r.elapsed_seconds)],['Evaluations',r=>value(r.n_evaluations)]],rows,r=>showCrn(r.topology_hash)); }
    async function loadLlm(){const rows=await api('/api/llm',{run_id:state.run});const root=$('llmRuns');const failed=rows.filter(row=>row.status==='failed').length;$('llmMeta').textContent=rows.length?`${rows.length} recorded batch${rows.length===1?'':'es'} · ${failed} failed`:'';if(!rows.length){root.innerHTML='<div class="empty">No LLM batches recorded yet</div>';return;}root.replaceChildren();rows.forEach((row,index)=>{const batch=document.createElement('button');batch.className=`llm-batch ${row.status==='failed'?'failed':''}`;batch.onclick=()=>showLlm(row);const epochs=document.createElement('div');epochs.className='epoch-span';const epochTitle=document.createElement('strong');epochTitle.textContent=`${row.launched_epoch} → ${row.completed_epoch}`;const epochLabel=document.createElement('span');epochLabel.textContent=`${row.completed_epoch-row.launched_epoch} RL epochs elapsed`;epochs.append(epochTitle,epochLabel);const title=document.createElement('div');title.className='batch-title';const titleText=document.createElement('strong');titleText.textContent=row.status==='failed'?'Failed proposal batch':`Proposal batch ${rows.length-index}`;const code=document.createElement('code');code.textContent=shortHash(row.llm_run_id);title.append(titleText,code);const validity=document.createElement('div');validity.className='batch-validity';const validityLine=document.createElement('div');validityLine.className='validity-line';const validText=document.createElement('span');validText.textContent=row.status==='failed'?`${row.rejected} rejected before evaluation`:`${row.valid_count}/${row.produced} valid`;const best=document.createElement('span');best.textContent=row.status==='failed'?'failed':`best ${fmt(row.best_loss)}`;validityLine.append(validText,best);const track=document.createElement('div');track.className='validity-track';const fill=document.createElement('span');fill.className='validity-fill';fill.style.width=`${row.produced?100*row.valid_count/row.produced:0}%`;track.appendChild(fill);validity.append(validityLine,track);const timing=document.createElement('div');timing.className='batch-stat';const timingValue=document.createElement('strong');timingValue.textContent=duration(row.elapsed_seconds);const timingLabel=document.createElement('span');timingLabel.textContent=`${row.requested} requested`;timing.append(timingValue,timingLabel);const open=document.createElement('div');open.className='batch-open';open.textContent='›';batch.append(epochs,title,validity,timing,open);root.appendChild(batch);});}
    async function loadCrns(){ const rows=await api('/api/crns',{run_id:state.run}); table('crnTable',[['Initial source',r=>value(r.initial_source,`source-${String(r.initial_source).toLowerCase()}`)],['Reaction IDs',r=>value(JSON.stringify(r.reaction_ids_json))],['Best loss',r=>value(r.best_loss,'good')],['Evaluations',r=>value(r.evaluation_count)],['Observed sources',r=>value(r.sources)],['Topology',r=>value(shortHash(r.topology_hash),'hash')],['Network',r=>value(`${r.reaction_ids_json.length} reactions`)]],rows,r=>showCrn(r.topology_hash)); }
    function speciesNode(name,outputs){const span=document.createElement('span');span.className='species'+(outputs.includes(name)?' output':'');const match=String(name).match(/^([A-Za-z]+)_([A-Za-z0-9]+)$/);if(match){span.append(document.createTextNode(match[1]));const sub=document.createElement('sub');sub.textContent=match[2];span.appendChild(sub)}else span.textContent=name;return span;}
    function sideNode(terms,outputs,product=false){const side=document.createElement('span');side.className='reaction-side'+(product?' products':'');if(!terms.length){side.textContent='∅';return side;}terms.forEach((term,index)=>{if(index){const plus=document.createElement('span');plus.className='plus';plus.textContent='+';side.appendChild(plus)}if(term.coefficient>1){const coefficient=document.createElement('span');coefficient.className='coefficient';coefficient.textContent=term.coefficient;side.appendChild(coefficient)}side.appendChild(speciesNode(term.species,outputs));});return side;}
    function reactionNode(reaction,outputs){const row=document.createElement('div');row.className=`reaction ${reaction.kind}`;const id=document.createElement('div');id.className='reaction-id';id.textContent=reaction.reaction_id===null?'R?':`R${reaction.reaction_id}`;const equation=document.createElement('div');equation.className='equation';equation.appendChild(sideNode(reaction.reactants,outputs));const arrow=document.createElement('span');arrow.className='reaction-arrow';const label=document.createElement('span');label.className='arrow-label';const annotations=[];if(reaction.rate!==null)annotations.push(`k=${fmt(reaction.rate)}`);annotations.push(...reaction.inputs);label.textContent=annotations.join(' · ');const line=document.createElement('span');line.className='arrow-line';arrow.append(label,line);equation.append(arrow,sideNode(reaction.products,outputs,true));const kind=document.createElement('div');kind.className='reaction-kind';kind.textContent=reaction.kind==='template'?'input-driven':'designed';row.append(id,equation,kind);return row;}
    function reactionSection(title,reactions,outputs){const section=document.createElement('div');section.className='reaction-section';const heading=document.createElement('h3');heading.textContent=title;const list=document.createElement('div');list.className='reaction-list';reactions.forEach(r=>list.appendChild(reactionNode(r,outputs)));section.append(heading,list);return section;}
    function appendInlineMarkdown(parent,text){String(text).split(/(\*\*.*?\*\*|`.*?`)/g).filter(Boolean).forEach(part=>{if(part.startsWith('**')&&part.endsWith('**')){const strong=document.createElement('strong');strong.textContent=part.slice(2,-2);parent.appendChild(strong)}else if(part.startsWith('`')&&part.endsWith('`')){const code=document.createElement('code');code.textContent=part.slice(1,-1);parent.appendChild(code)}else parent.appendChild(document.createTextNode(part));});}
    function reasoningContent(text){const root=document.createElement('div');let list=null;String(text).split('\n').forEach(raw=>{const line=raw.trim();if(!line){list=null;return;}if(line.startsWith('- ')){if(!list){list=document.createElement('ul');root.appendChild(list)}const item=document.createElement('li');appendInlineMarkdown(item,line.slice(2));list.appendChild(item);return;}list=null;if(line.startsWith('### ')){const heading=document.createElement('h4');heading.textContent=line.slice(4);root.appendChild(heading);return;}const paragraph=document.createElement('p');appendInlineMarkdown(paragraph,line);root.appendChild(paragraph);});return root;}
    function reasoningSection(row){const section=document.createElement('div');section.className='reasoning-section';const heading=document.createElement('h3');heading.textContent='Reasoning summary by request';section.appendChild(heading);const requests=row.reasoning_requests||[];if(!requests.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No workspace reasoning artifact is available for this batch.';section.appendChild(empty);}else requests.forEach(request=>{const block=document.createElement('div');block.className='reasoning-request';const title=document.createElement('h4');title.textContent=request.title;block.append(title,reasoningContent(request.content));section.appendChild(block)});if(row.reasoning_workspace){const path=document.createElement('span');path.className='reasoning-path';path.textContent=row.reasoning_workspace;section.appendChild(path)}return section;}
    async function showCrn(hash){const d=await api('/api/crn',{topology_hash:hash});const p=d.presentation||{inputs:[],species:[],outputs:[],reactions:[],latex:''};$('detailTitle').textContent=`CRN ${shortHash(hash)}`;const body=$('detailBody');body.replaceChildren();const meta=document.createElement('div');meta.className='crn-meta';[['Initial source',d.initial_source],['Inputs',p.inputs],['Species',p.species],['Outputs',p.outputs],['Reactions',p.reactions.length]].forEach(([label,data])=>{const chip=document.createElement('span');chip.className='meta-chip';if(label==='Initial source')chip.classList.add(`source-${String(data).toLowerCase()}`);chip.textContent=`${label}: ${Array.isArray(data)?data.join(', '):data}`;meta.appendChild(chip)});body.appendChild(meta);if(d.has_hof_plot){const figure=document.createElement('figure');figure.className='trajectory-figure';const caption=document.createElement('figcaption');caption.textContent='Saved Hall-of-Fame trajectories';const image=document.createElement('img');image.src=assetUrl('/api/hof-plot',{topology_hash:hash});image.alt='Cached simulation trajectories for this Hall-of-Fame CRN';image.loading='lazy';figure.append(caption,image);body.appendChild(figure)}const template=p.reactions.filter(r=>r.kind==='template'),designed=p.reactions.filter(r=>r.kind!=='template');if(template.length)body.appendChild(reactionSection('Fixed template reactions',template,p.outputs));if(designed.length)body.appendChild(reactionSection('Designed reactions',designed,p.outputs));const latex=document.createElement('div');latex.className='latex-box';const latexHead=document.createElement('div');latexHead.className='latex-head';const label=document.createElement('strong');label.textContent='LaTeX';const copy=document.createElement('button');copy.textContent='Copy LaTeX';copy.onclick=async()=>{await navigator.clipboard.writeText(p.latex);copy.textContent='Copied';setTimeout(()=>copy.textContent='Copy LaTeX',1200)};latexHead.append(label,copy);const code=document.createElement('pre');code.textContent=p.latex;latex.append(latexHead,code);body.appendChild(latex);const raw=document.createElement('details');raw.className='raw-record';const summary=document.createElement('summary');summary.textContent='Raw database record';const pre=document.createElement('pre');pre.textContent=JSON.stringify(d,null,2);raw.append(summary,pre);body.appendChild(raw);if(!$('detail').open)$('detail').showModal();}
    async function showLlm(row){const candidates=await api('/api/llm-candidates',{llm_run_id:row.llm_run_id});$('detailTitle').textContent=`LLM proposal batch · epoch ${row.launched_epoch}`;const body=$('detailBody');body.replaceChildren();const meta=document.createElement('div');meta.className='crn-meta';[['Status',row.status],['Epochs',`${row.launched_epoch} → ${row.completed_epoch}`],['Duration',duration(row.elapsed_seconds)],['Accepted',`${row.accepted??row.valid_count}/${row.returned??row.produced}`],['Rejected',row.rejected??0],['Clamped values',row.clamped_parameters??'-'],['Best loss',fmt(row.best_loss)]].forEach(([label,data])=>{const chip=document.createElement('span');chip.className='meta-chip';chip.textContent=`${label}: ${data}`;meta.appendChild(chip)});body.append(meta,reasoningSection(row));if(row.error){const error=document.createElement('div');error.className='error';error.textContent=row.error;body.appendChild(error);}const heading=document.createElement('div');heading.className='reaction-section';const h=document.createElement('h3');h.textContent='Candidates';const list=document.createElement('div');list.className='candidate-list';if(!candidates.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No candidates reached canonical evaluation in this batch.';list.appendChild(empty);}candidates.forEach(candidate=>{const item=document.createElement('div');item.className=`candidate-row ${candidate.valid?'':'invalid'}`;const index=document.createElement('div');index.className='candidate-index';index.textContent=`#${candidate.candidate_index+1}`;const main=document.createElement('div');main.className='candidate-main';const ids=document.createElement('strong');const reactionIds=candidate.reaction_ids_json||candidate.candidate_json.reaction_ids||[];ids.textContent=`IDs ${reactionIds.join(', ')}`;const params=document.createElement('span');const values=(candidate.candidate_json.parameter_values||[]).flat();params.textContent=`k = ${values.map(fmt).join(', ')}`;main.append(ids,params);const result=document.createElement('div');const status=document.createElement('span');status.className=`status-pill ${candidate.valid?'valid':'invalid'}`;status.textContent=candidate.valid?'valid':'invalid';const loss=document.createElement('div');loss.className=candidate.valid?'good':'bad';loss.textContent=`loss ${fmt(candidate.loss)}`;result.append(status,loss);const inspect=document.createElement('button');inspect.textContent=candidate.topology_hash?'View network':'Unavailable';inspect.disabled=!candidate.topology_hash;inspect.onclick=()=>candidate.topology_hash&&showCrn(candidate.topology_hash);item.append(index,main,result,inspect);list.appendChild(item)});heading.append(h,list);body.appendChild(heading);const raw=document.createElement('details');raw.className='raw-record';const summary=document.createElement('summary');summary.textContent='Raw batch record';const pre=document.createElement('pre');pre.textContent=JSON.stringify({batch:row,candidates},null,2);raw.append(summary,pre);body.appendChild(raw);if(!$('detail').open)$('detail').showModal();}
    function showDetail(title,data){$('detailTitle').textContent=title;const pre=document.createElement('pre');pre.textContent=JSON.stringify(data,null,2);$('detailBody').replaceChildren(pre);if(!$('detail').open)$('detail').showModal();}
    async function load(){ showError(null);$('status').textContent='Loading...';try{ if(state.view==='campaigns')await loadCampaigns();if(state.view==='overview')await loadOverview();if(state.view==='hof')await loadHof();if(state.view==='optimizations')await loadOptimizations();if(state.view==='llm')await loadLlm();if(state.view==='crns')await loadCrns();$('status').textContent='';}catch(e){showError(e);$('status').textContent='';} }
    document.querySelectorAll('nav button,main section').forEach(x=>x.classList.remove('active'));document.querySelector(`nav button[data-view="${state.view}"]`).classList.add('active');$(state.view).classList.add('active');
    $('tabs').onclick=e=>{const b=e.target.closest('button[data-view]');if(!b)return;state.view=b.dataset.view;document.querySelectorAll('nav button,main section').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(state.view).classList.add('active');load();};
    $('taskSelect').onchange=e=>{state.task=e.target.value;renderRuns();load();};$('runSelect').onchange=e=>{state.run=e.target.value;load();};$('refresh').onclick=load;$('closeDetail').onclick=()=>$('detail').close();window.onresize=()=>{if(state.view==='overview')drawChart()};
    loadRuns().then(async()=>{const summary=await api('/api/summary',{run_id:state.run});$('dbPath').textContent=`Trial ${summary.trial_id} | ${summary.database}`;await load();}).catch(showError);
    setInterval(()=>{if(state.view==='campaigns')load();},15000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
