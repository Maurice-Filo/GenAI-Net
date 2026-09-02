"""Persistent, non-blocking storage for CRN search results."""

from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


SCHEMA_VERSION = 3


class ResultsDatabase:
    """Write CRN results to SQLite through a single background thread.

    Objects from the live training session are serialized before they enter the
    queue.  The writer therefore never retains or mutates environments, CRNs,
    tasks, or trainer state.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: Optional[str] = None,
        run_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id or uuid.uuid4())
        self._queue: queue.Queue[Any] = queue.Queue()
        self._error: Optional[BaseException] = None
        self._closed = False
        self._initialize_schema()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"rl4crn-results-{self.run_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        self._enqueue(
            "run",
            {
                "run_id": self.run_id,
                "created_at": time.time(),
                "metadata_json": _json_dumps(dict(run_metadata or {})),
            },
        )

    def record_evaluation(
        self,
        state: Any,
        *,
        source: str,
        epoch: Optional[int],
        loss: Optional[float] = None,
        valid: bool = True,
        message: str = "",
        task_info: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Queue one evaluated parameterization for durable storage."""

        crn = serialize_crn(state)
        info = dict(task_info or getattr(state, "last_task_info", {}) or {})
        if loss is None:
            loss = _optional_float(info.get("reward"))
        self._enqueue(
            "evaluation",
            {
                "run_id": self.run_id,
                "crn": crn,
                "source": str(source),
                "epoch": None if epoch is None else int(epoch),
                "loss": _optional_float(loss),
                "valid": int(bool(valid)),
                "message": str(message),
                "parameters_json": crn["parameters_json"],
                "task_info_json": _task_info_dumps(info),
                "metadata_json": _json_dumps(dict(metadata or {})),
                "created_at": time.time(),
            },
        )

    def record_llm_round(
        self,
        result: Any,
        *,
        launched_epoch: int,
        completed_epoch: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        requested: Optional[int] = None,
    ) -> None:
        """Queue an LLM round and each candidate evaluation it produced."""

        evaluations = list(getattr(result, "evaluations", []) or [])
        validation = dict(getattr(result, "response_validation", {}) or {})
        llm_run_id = str(uuid.uuid4())
        self._enqueue(
            "llm_run",
            {
                "llm_run_id": llm_run_id,
                "run_id": self.run_id,
                "launched_epoch": int(launched_epoch),
                "completed_epoch": int(
                    launched_epoch if completed_epoch is None else completed_epoch
                ),
                "requested": None if requested is None else int(requested),
                "produced": len(evaluations),
                "valid_count": sum(bool(getattr(ev, "valid", False)) for ev in evaluations),
                "returned": _optional_int(validation.get("returned_candidate_count")),
                "accepted": _optional_int(validation.get("accepted_candidate_count")),
                "rejected": len(validation.get("rejected_candidates", ()) or ()),
                "clamped_parameters": int(
                    validation.get("clamped_parameter_count", 0) or 0
                ),
                "provider_call_count": _optional_int(
                    validation.get("provider_call_count")
                ),
                "response_validation_json": _json_dumps(validation),
                "elapsed_seconds": _optional_float(elapsed_seconds),
                "created_at": time.time(),
            },
        )
        for index, evaluation in enumerate(evaluations):
            env = getattr(evaluation, "env", None)
            candidate = getattr(evaluation, "candidate", None)
            candidate_payload = (
                candidate.to_dict() if candidate is not None and hasattr(candidate, "to_dict") else {}
            )
            crn = serialize_crn(env.state) if env is not None and hasattr(env, "state") else None
            self._enqueue(
                "llm_candidate",
                {
                    "llm_run_id": llm_run_id,
                    "candidate_index": index,
                    "crn": crn,
                    "candidate_json": _json_dumps(candidate_payload),
                    "valid": int(bool(getattr(evaluation, "valid", False))),
                    "loss": _optional_float(getattr(evaluation, "loss", None)),
                    "message": str(getattr(evaluation, "message", "")),
                    "task_info_json": _task_info_dumps(
                        getattr(evaluation, "task_info", {}) or {}
                    ),
                },
            )
            if crn is None:
                continue
            self.record_evaluation(
                env.state,
                source="llm",
                epoch=launched_epoch,
                loss=getattr(evaluation, "loss", None),
                valid=bool(getattr(evaluation, "valid", False)),
                message=str(getattr(evaluation, "message", "")),
                task_info=getattr(evaluation, "task_info", {}) or {},
                metadata={
                    "llm_run_id": llm_run_id,
                    "candidate_index": index,
                    "candidate": candidate_payload,
                },
            )

    def record_llm_failure(
        self,
        *,
        launched_epoch: int,
        completed_epoch: int,
        requested: int,
        elapsed_seconds: float,
        error: str,
        response_validation: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Queue one failed model round without inventing candidate evaluations."""

        validation = dict(response_validation or {})
        self._enqueue(
            "llm_failure",
            {
                "failure_id": str(uuid.uuid4()),
                "run_id": self.run_id,
                "launched_epoch": int(launched_epoch),
                "completed_epoch": int(completed_epoch),
                "requested": int(requested),
                "returned": _optional_int(validation.get("returned_candidate_count")),
                "accepted": _optional_int(validation.get("accepted_candidate_count")),
                "rejected": len(validation.get("rejected_candidates", ()) or ()),
                "clamped_parameters": int(
                    validation.get("clamped_parameter_count", 0) or 0
                ),
                "elapsed_seconds": float(elapsed_seconds),
                "error": str(error),
                "response_validation_json": _json_dumps(validation),
                "created_at": time.time(),
            },
        )

    def record_optimization(
        self,
        original_state: Any,
        optimized_state: Any,
        *,
        epoch: int,
        rank: int,
        original_loss: float,
        optimized_loss: float,
        attempted: bool,
        success: bool,
        message: str,
        elapsed_seconds: float,
        n_evaluations: int,
        stored: bool,
    ) -> None:
        """Queue the complete outcome of one fixed-topology optimization."""

        original = serialize_crn(original_state)
        optimized = serialize_crn(optimized_state)
        self._enqueue(
            "optimization",
            {
                "optimization_id": str(uuid.uuid4()),
                "run_id": self.run_id,
                "crn": optimized,
                "epoch": int(epoch),
                "hof_rank": int(rank),
                "original_loss": float(original_loss),
                "optimized_loss": float(optimized_loss),
                "attempted": int(bool(attempted)),
                "success": int(bool(success)),
                "stored": int(bool(stored)),
                "message": str(message),
                "elapsed_seconds": float(elapsed_seconds),
                "n_evaluations": int(n_evaluations),
                "original_parameters_json": original["parameters_json"],
                "optimized_parameters_json": optimized["parameters_json"],
                "created_at": time.time(),
            },
        )

    def record_hof_snapshot(
        self,
        environments: Iterable[Any],
        *,
        epoch: int,
        save_plots: bool = False,
        llm_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        """Queue a ranked, point-in-time Hall-of-Fame snapshot."""

        entries = []
        for rank, env in enumerate(environments):
            state = getattr(env, "state", env)
            crn = serialize_crn(state)
            info = dict(getattr(state, "last_task_info", {}) or {})
            provenance = classify_hof_provenance(
                crn,
                info,
                llm_provenance or {},
            )
            info.update(provenance)
            entries.append(
                {
                    "rank": rank,
                    "crn": crn,
                    "loss": _optional_float(info.get("reward")),
                    "parameters_json": crn["parameters_json"],
                    "task_info_json": _task_info_dumps(info),
                    **provenance,
                    "plot_outputs": _copy_plot_outputs(info.get("outputs", ()))
                    if save_plots
                    else [],
                    "plot_path": str(
                        self.path.parent / "hof-plots" / f"{crn['topology_hash']}.jpg"
                    ),
                }
            )
        self._enqueue(
            "hof_snapshot",
            {
                "snapshot_id": str(uuid.uuid4()),
                "run_id": self.run_id,
                "epoch": int(epoch),
                "created_at": time.time(),
                "entries": entries,
            },
        )

    def flush(self) -> None:
        """Wait until queued writes are durable and surface writer failures."""

        self._queue.join()
        self._raise_if_failed()

    def close(self) -> None:
        """Flush pending records and stop the writer thread."""

        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self._raise_if_failed()

    def _enqueue(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("ResultsDatabase is closed.")
        self._raise_if_failed()
        self._queue.put((kind, payload))

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Results database writer failed.") from self._error

    def _initialize_schema(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS crns (
                    topology_hash TEXT PRIMARY KEY,
                    reaction_ids_json TEXT NOT NULL,
                    structure_json TEXT NOT NULL,
                    crn_text TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    topology_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    epoch INTEGER,
                    loss REAL,
                    valid INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    task_info_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES training_runs(run_id),
                    FOREIGN KEY(topology_hash) REFERENCES crns(topology_hash)
                );
                CREATE TABLE IF NOT EXISTS llm_runs (
                    llm_run_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    launched_epoch INTEGER NOT NULL,
                    completed_epoch INTEGER NOT NULL,
                    requested INTEGER,
                    produced INTEGER NOT NULL,
                    valid_count INTEGER NOT NULL,
                    returned INTEGER,
                    accepted INTEGER,
                    rejected INTEGER NOT NULL,
                    clamped_parameters INTEGER NOT NULL,
                    provider_call_count INTEGER,
                    response_validation_json TEXT NOT NULL,
                    elapsed_seconds REAL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES training_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS llm_candidates (
                    llm_run_id TEXT NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    topology_hash TEXT,
                    candidate_json TEXT NOT NULL,
                    valid INTEGER NOT NULL,
                    loss REAL,
                    message TEXT NOT NULL,
                    task_info_json TEXT NOT NULL,
                    PRIMARY KEY(llm_run_id, candidate_index),
                    FOREIGN KEY(llm_run_id) REFERENCES llm_runs(llm_run_id),
                    FOREIGN KEY(topology_hash) REFERENCES crns(topology_hash)
                );
                CREATE TABLE IF NOT EXISTS llm_failures (
                    failure_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    launched_epoch INTEGER NOT NULL,
                    completed_epoch INTEGER NOT NULL,
                    requested INTEGER NOT NULL,
                    returned INTEGER,
                    accepted INTEGER,
                    rejected INTEGER NOT NULL,
                    clamped_parameters INTEGER NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    error TEXT NOT NULL,
                    response_validation_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES training_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    optimization_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    topology_hash TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    hof_rank INTEGER NOT NULL,
                    original_loss REAL NOT NULL,
                    optimized_loss REAL NOT NULL,
                    attempted INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    stored INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    n_evaluations INTEGER NOT NULL,
                    original_parameters_json TEXT NOT NULL,
                    optimized_parameters_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES training_runs(run_id),
                    FOREIGN KEY(topology_hash) REFERENCES crns(topology_hash)
                );
                CREATE TABLE IF NOT EXISTS hof_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES training_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS hof_snapshot_entries (
                    snapshot_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    topology_hash TEXT NOT NULL,
                    loss REAL,
                    parameters_json TEXT NOT NULL,
                    task_info_json TEXT NOT NULL,
                    emitter TEXT NOT NULL,
                    provenance_class TEXT NOT NULL,
                    related_llm_proposal_id TEXT,
                    related_llm_first_seen_epoch INTEGER,
                    PRIMARY KEY(snapshot_id, rank),
                    FOREIGN KEY(snapshot_id) REFERENCES hof_snapshots(snapshot_id),
                    FOREIGN KEY(topology_hash) REFERENCES crns(topology_hash)
                );
                CREATE INDEX IF NOT EXISTS evaluations_topology_idx
                    ON evaluations(topology_hash, loss);
                CREATE INDEX IF NOT EXISTS optimization_topology_idx
                    ON optimization_runs(topology_hash, optimized_loss);
                CREATE INDEX IF NOT EXISTS snapshots_run_epoch_idx
                    ON hof_snapshots(run_id, epoch);
                """
            )
            row = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported results database schema {row[0]}; expected {SCHEMA_VERSION}."
                )

    def _writer_loop(self) -> None:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    kind, payload = item
                    self._write(connection, kind, payload)
                    connection.commit()
                except BaseException as exc:
                    connection.rollback()
                    self._error = exc
                finally:
                    self._queue.task_done()
        finally:
            connection.close()

    @staticmethod
    def _write(connection: sqlite3.Connection, kind: str, p: Dict[str, Any]) -> None:
        if kind == "run":
            connection.execute(
                "INSERT OR IGNORE INTO training_runs VALUES (?, ?, ?)",
                (p["run_id"], p["created_at"], p["metadata_json"]),
            )
            return
        if kind in {"evaluation", "optimization"}:
            _insert_crn(connection, p["crn"])
        if kind == "evaluation":
            connection.execute(
                """INSERT INTO evaluations(
                    run_id, topology_hash, source, epoch, loss, valid, message,
                    parameters_json, task_info_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    p["run_id"], p["crn"]["topology_hash"], p["source"], p["epoch"],
                    p["loss"], p["valid"], p["message"], p["parameters_json"],
                    p["task_info_json"], p["metadata_json"], p["created_at"],
                ),
            )
            return
        if kind == "llm_run":
            connection.execute(
                "INSERT INTO llm_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(p[key] for key in (
                    "llm_run_id", "run_id", "launched_epoch", "completed_epoch",
                    "requested", "produced", "valid_count", "returned", "accepted",
                    "rejected", "clamped_parameters", "provider_call_count",
                    "response_validation_json", "elapsed_seconds", "created_at",
                )),
            )
            return
        if kind == "llm_candidate":
            if p["crn"] is not None:
                _insert_crn(connection, p["crn"])
            connection.execute(
                "INSERT INTO llm_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    p["llm_run_id"], p["candidate_index"],
                    None if p["crn"] is None else p["crn"]["topology_hash"],
                    p["candidate_json"], p["valid"], p["loss"], p["message"],
                    p["task_info_json"],
                ),
            )
            return
        if kind == "llm_failure":
            connection.execute(
                "INSERT INTO llm_failures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(
                    p[key]
                    for key in (
                        "failure_id",
                        "run_id",
                        "launched_epoch",
                        "completed_epoch",
                        "requested",
                        "returned",
                        "accepted",
                        "rejected",
                        "clamped_parameters",
                        "elapsed_seconds",
                        "error",
                        "response_validation_json",
                        "created_at",
                    )
                ),
            )
            return
        if kind == "optimization":
            connection.execute(
                """INSERT INTO optimization_runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    p["optimization_id"], p["run_id"], p["crn"]["topology_hash"],
                    p["epoch"], p["hof_rank"], p["original_loss"], p["optimized_loss"],
                    p["attempted"], p["success"], p["stored"], p["message"],
                    p["elapsed_seconds"], p["n_evaluations"], p["original_parameters_json"],
                    p["optimized_parameters_json"], p["created_at"],
                ),
            )
            return
        if kind == "hof_snapshot":
            connection.execute(
                "INSERT INTO hof_snapshots VALUES (?, ?, ?, ?)",
                (p["snapshot_id"], p["run_id"], p["epoch"], p["created_at"]),
            )
            for entry in p["entries"]:
                _insert_crn(connection, entry["crn"])
                connection.execute(
                    """INSERT INTO hof_snapshot_entries(
                        snapshot_id, rank, topology_hash, loss, parameters_json,
                        task_info_json, emitter, provenance_class,
                        related_llm_proposal_id, related_llm_first_seen_epoch
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        p["snapshot_id"], entry["rank"], entry["crn"]["topology_hash"],
                        entry["loss"], entry["parameters_json"], entry["task_info_json"],
                        entry["emitter"], entry["provenance_class"],
                        entry.get("related_llm_proposal_id"),
                        entry.get("related_llm_first_seen_epoch"),
                    ),
                )
                if entry.get("plot_outputs"):
                    _write_hof_plot(
                        Path(entry["plot_path"]),
                        entry["plot_outputs"],
                        loss=entry.get("loss"),
                    )
            return
        raise ValueError(f"Unknown results database record kind: {kind}")


def serialize_crn(state: Any) -> Dict[str, Any]:
    """Return stable topology identity plus JSON-safe CRN details."""

    reactions = []
    parameterized_topology = []
    topology = []
    for index, reaction in enumerate(getattr(state, "reactions", []) or []):
        reaction_id = getattr(reaction, "ID", None)
        structural = {
            "type": type(reaction).__name__,
            "reactants": list(getattr(reaction, "reactant_labels", []) or []),
            "products": list(getattr(reaction, "product_labels", []) or []),
            "inputs": list(getattr(reaction, "input_channels", []) or []),
        }
        topology_key = {"reaction_id": reaction_id, "structure": structural}
        topology.append(topology_key)
        reactions.append(
            {
                "index": index,
                **topology_key,
                "parameters": list(getattr(reaction, "params", []) or []),
            }
        )
        parameterized_topology.append(
            {
                **topology_key,
                "parameters": list(getattr(reaction, "params", []) or []),
            }
        )
    topology.sort(key=lambda item: _json_dumps(item))
    topology_json = _json_dumps(topology)
    parameterized_topology.sort(key=lambda item: _json_dumps(item))
    parameterization_json = _json_dumps(parameterized_topology)
    reaction_ids = sorted(
        int(item["reaction_id"])
        for item in topology
        if item["reaction_id"] is not None
    )
    return {
        "topology_hash": hashlib.sha256(topology_json.encode("utf-8")).hexdigest(),
        "candidate_hash": hashlib.sha256(parameterization_json.encode("utf-8")).hexdigest(),
        "reaction_ids_json": _json_dumps(reaction_ids),
        "structure_json": topology_json,
        "parameters_json": _json_dumps(reactions),
        "crn_text": str(state),
        "created_at": time.time(),
    }


def classify_hof_provenance(
    crn: Mapping[str, Any],
    task_info: Mapping[str, Any],
    llm_provenance: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Classify emitter and prior-LLM structural relationships without causal inference."""

    emitter = str(task_info.get("emitter", task_info.get("source", "RL"))).upper()
    if emitter == "LLM":
        return {
            "emitter": "LLM",
            "provenance_class": "direct_llm",
            "related_llm_proposal_id": task_info.get("llm_proposal_id"),
            "related_llm_first_seen_epoch": task_info.get("llm_first_seen_epoch"),
        }

    record = llm_provenance.get(str(crn["topology_hash"]))
    if (
        not record
        or not bool(record.get("exposed_to_rl", False))
        or str(record.get("topology_first_emitter", "RL")).upper() != "LLM"
    ):
        return {
            "emitter": "RL",
            "provenance_class": "rl_native_topology",
            "related_llm_proposal_id": None,
            "related_llm_first_seen_epoch": None,
        }

    exact_hashes = set(record.get("candidate_hashes", ()) or ())
    provenance_class = (
        "rl_exact_reemission_of_llm_candidate"
        if crn.get("candidate_hash") in exact_hashes
        else "rl_parameter_refinement_of_llm_topology"
    )
    return {
        "emitter": "RL",
        "provenance_class": provenance_class,
        "related_llm_proposal_id": record.get("first_proposal_id"),
        "related_llm_first_seen_epoch": record.get("first_seen_epoch"),
    }


def _copy_plot_outputs(outputs: Iterable[Any]) -> list[np.ndarray]:
    copied = []
    for output in outputs or ():
        try:
            array = np.asarray(output, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size:
            copied.append(array.copy())
    return copied


def _write_hof_plot(path: Path, outputs: Iterable[Any], *, loss: Any) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        lines = 0
        for scenario, output in enumerate(outputs):
            array = np.asarray(output, dtype=float)
            rows = array.reshape((-1, array.shape[-1] if array.ndim > 1 else array.size))
            for output_index, row in enumerate(rows):
                if lines >= 32:
                    break
                axis.plot(row, alpha=0.7, label=f"scenario {scenario}, output {output_index}")
                lines += 1
        if not lines:
            plt.close(figure)
            return
        axis.set_xlabel("time index")
        axis.set_ylabel("output")
        axis.set_title(f"Hall-of-Fame trajectories, loss={loss}")
        if lines <= 12:
            axis.legend(fontsize=7)
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.jpg")
        figure.savefig(temporary, format="jpeg", dpi=140, facecolor="white")
        plt.close(figure)
        temporary.replace(path)
    except Exception:
        return


def _insert_crn(connection: sqlite3.Connection, crn: Dict[str, Any]) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO crns VALUES (?, ?, ?, ?, ?)",
        (
            crn["topology_hash"], crn["reaction_ids_json"], crn["structure_json"],
            crn["crn_text"], crn["created_at"],
        ),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _task_info_dumps(value: Any) -> str:
    return json.dumps(
        _compact_task_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _compact_task_value(value: Any, *, max_items: int = 128) -> Any:
    """Keep metrics intact while avoiding repeated storage of full trajectories."""

    if isinstance(value, Mapping):
        return {
            str(key): _compact_task_value(item, max_items=max_items)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        if len(value) > max_items:
            return {"__sequence_summary__": {"length": len(value)}}
        return [_compact_task_value(item, max_items=max_items) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    size = getattr(value, "size", None)
    if isinstance(size, int) and size > max_items:
        return {
            "__array_summary__": {
                "shape": list(getattr(value, "shape", (size,))),
                "dtype": str(getattr(value, "dtype", type(value).__name__)),
            }
        }
    if hasattr(value, "numel") and callable(value.numel) and int(value.numel()) > max_items:
        return {
            "__array_summary__": {
                "shape": list(getattr(value, "shape", (int(value.numel()),))),
                "dtype": str(getattr(value, "dtype", type(value).__name__)),
            }
        }
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)
