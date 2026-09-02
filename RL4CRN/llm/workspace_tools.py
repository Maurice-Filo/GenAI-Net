"""Capped, workspace-local evaluation tools for DeepSeek Harness agents."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from RL4CRN.llm.schemas import LLMCandidate


CRN_SIMULATION_SKILL = """---
name: crn-simulation
description: Evaluate a proposed CRN with the benchmark's CVODE evaluator and inspect compact diagnostics.
whenToUse: Use sparingly when simulation evidence could distinguish candidate mechanisms or parameter choices.
---

# CRN simulation

Use only the supplied workspace file queue; do not invoke Bash, install software, or import
project code. The evaluator runs in a separate process owned by the benchmark.

First inspect the cached Hall-of-Fame diagnostics and plots under `CONTEXT/hall-of-fame/`.
Those files were extracted from simulations already performed by the live run and cost no new
evaluations. Use the evaluator queue only when cached evidence cannot answer the question.

The default is **do not run an exploratory simulation**. Propose now, then learn from the external
evaluation recorded in the next scheduled workspace. Use this queue only when you can name a
specific uncertainty and explain how its result would change which candidates you submit.

1. Select at most three high-information candidates. Prefer fewer. Do not pre-evaluate every final
   proposal or use simulation simply because evaluation budget remains.
2. With the workspace file-writing tool, create exactly one fresh request named
   `tool-requests/probe-batch-01.request.json` as:
   `{"candidates":[{"reaction_ids":[...],"parameter_values":[...]}, ...]}`.
3. With the workspace file-reading tool, read the matching
   `tool-requests/probe-batch-01.response.json`. If it is not present yet, retry the read;
   do not use a shell command to wait.
4. Read the `evaluations` array containing loss, validity, and diagnostic artifact paths.
5. Inspect the PNG plot if image viewing is available. Otherwise read the corresponding
   `diagnostics.json`, which reports trajectory shapes, ranges, and terminal values.
6. Record the evidence used in `REASONING_NOTES.md`.

The evaluation allowance is capped and every call is logged for benchmark accounting.
Do not treat exploratory simulations as free evaluations, and do not claim success from an
unevaluated candidate. Never report that the evaluator was unavailable merely because Bash is
sandboxed: this protocol deliberately uses workspace file operations instead.
"""


REASONING_NOTES_TEMPLATE = """# CRN Proposal Notes

## Search approach

Briefly summarize the mechanisms and parameter regimes considered.

## Evidence consulted

List relevant Hall-of-Fame entries, excluded topologies, and optional evaluator calls. State
clearly when a claim is only a hypothesis.

## Final selection

Summarize why the proposed candidates were retained, including the exploration/refinement mix.
Do not include private chain-of-thought; record concise scientific rationale and observable evidence.
"""


LITERATURE_SEARCH_SKILL = """---
name: synthetic-biology-literature-search
description: Search the experiment's local open-access synthetic-biology paper index.
whenToUse: Use when a concrete mechanistic precedent could improve the proposed CRNs.
---

# Literature search

This optional tool searches a fixed, read-only corpus. It does not browse the internet.

1. First inspect the task and cached run evidence. Search only for a concrete mechanistic question.
2. Create `literature-requests/search-01.request.json` with
   `{"query":"specific terms","limit":5}` using workspace file writing.
3. Read `literature-requests/search-01.response.json`. If absent, retry the read without Bash.
4. Cite paper identifiers in `REASONING_NOTES.md` when a result influenced a candidate.

Use at most the number of searches stated in `CONTEXT/literature_endpoint.json`. The retrieved
passages are evidence, not benchmark outcomes; the external CVODE evaluator remains authoritative.
"""


def default_workspace_tool_files(*, include_literature: bool = False) -> Dict[str, str]:
    """Return the project-local evaluation skill and reasoning-note files."""

    files = {
        ".dsh/skills/crn-simulation/SKILL.md": CRN_SIMULATION_SKILL,
        "REASONING_NOTES.md": REASONING_NOTES_TEMPLATE,
    }
    if include_literature:
        files[".dsh/skills/literature-search/SKILL.md"] = LITERATURE_SEARCH_SKILL
    return files


class WorkspaceLiteratureService:
    """Serve capped searches from a fixed SQLite corpus through workspace files."""

    def __init__(
        self,
        workspace: Path,
        database: Path,
        *,
        max_searches: int = 2,
        default_limit: int = 5,
    ) -> None:
        self.workspace = Path(workspace)
        self.database = Path(database).expanduser().resolve()
        self.max_searches = int(max_searches)
        self.default_limit = int(default_limit)
        if not self.database.is_file():
            raise FileNotFoundError(self.database)
        if self.max_searches < 0 or self.default_limit <= 0:
            raise ValueError("literature search limits are invalid")
        self.records: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.queue_path = self.workspace / "literature-requests"

    def __enter__(self) -> "WorkspaceLiteratureService":
        self.queue_path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.queue_path, 0o700)
        _write_json(
            self.workspace / "CONTEXT/literature_endpoint.json",
            {
                "transport": "workspace-file-queue",
                "queue": str(self.queue_path.relative_to(self.workspace)),
                "corpus": "fixed open-access synthetic-biology corpus",
                "maximum_searches": self.max_searches,
                "default_results": self.default_limit,
            },
        )
        self._thread = threading.Thread(
            target=self._serve,
            name="rl4crn-literature-search",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        _write_json(
            self.workspace / "literature_search_summary.json",
            {"used": len(self.records), "maximum": self.max_searches},
        )

    def _serve(self) -> None:
        from literature_rag.search import search_database

        while not self._stop.is_set():
            handled = False
            for request_path in sorted(self.queue_path.glob("*.request.json")):
                response_path = request_path.with_name(
                    request_path.name.replace(".request.json", ".response.json")
                )
                if response_path.exists():
                    continue
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if len(self.records) >= self.max_searches:
                    response = {"valid": False, "error": "literature search allowance exhausted"}
                else:
                    try:
                        query = str(request.get("query", "")).strip()
                        limit = min(int(request.get("limit", self.default_limit)), 10)
                        results = search_database(self.database, query, limit=limit)
                        response = {"valid": True, "query": query, "results": results}
                        self.records.append(
                            {"query": query, "result_count": len(results), "request": request_path.name}
                        )
                    except Exception as exc:
                        response = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
                response["used"] = len(self.records)
                response["maximum"] = self.max_searches
                _write_json(response_path, response)
                handled = True
            if not handled:
                self._stop.wait(0.05)


class WorkspaceEvaluationService:
    """Expose one evaluator through an auditable workspace file queue with a hard cap."""

    def __init__(self, workspace: Path, evaluator: Any, *, max_evaluations: int = 10):
        self.workspace = Path(workspace)
        self.evaluator = evaluator
        self.max_evaluations = int(max_evaluations)
        if self.max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative")
        self.records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.queue_path = self.workspace / "tool-requests"

    def __enter__(self) -> "WorkspaceEvaluationService":
        self.queue_path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.queue_path, 0o700)
        self._thread = threading.Thread(
            target=self._serve,
            name="rl4crn-workspace-evaluator",
            daemon=True,
        )
        self._thread.start()
        endpoint = {
            "transport": "workspace-file-queue",
            "queue": str(self.queue_path.relative_to(self.workspace)),
            "timeout_seconds": 300,
            "maximum_evaluations": self.max_evaluations,
        }
        _write_json(self.workspace / "CONTEXT/evaluator_endpoint.json", endpoint)
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        _write_json(
            self.workspace / "tool_evaluation_summary.json",
            {"used": len(self.records), "maximum": self.max_evaluations},
        )

    def _serve(self) -> None:
        while not self._stop.is_set():
            handled = False
            for request_path in sorted(self.queue_path.glob("*.request.json")):
                response_path = request_path.with_name(
                    request_path.name.replace(".request.json", ".response.json")
                )
                if response_path.exists():
                    continue
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    # A filesystem editor may expose the path just before its write completes.
                    continue
                try:
                    response = self._evaluate_request(request)
                except Exception as exc:
                    response = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
                _write_json(response_path, response)
                handled = True
            if not handled:
                self._stop.wait(0.05)

    def _evaluate_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        if "candidates" not in request:
            return self.evaluate(request.get("candidate", {}))
        candidates = request["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("batch evaluation requires a non-empty candidates list")
        evaluations = []
        for candidate in candidates:
            try:
                evaluations.append(self.evaluate(candidate))
            except Exception as exc:
                evaluations.append({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        return {
            "evaluations": evaluations,
            "used": len(self.records),
            "maximum": self.max_evaluations,
        }

    def evaluate(self, raw_candidate: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            index = len(self.records) + 1
            if len(self.records) >= self.max_evaluations:
                return {
                    "valid": False,
                    "error": "workspace evaluation allowance exhausted",
                    "used": len(self.records),
                    "maximum": self.max_evaluations,
                }
            candidate = LLMCandidate.from_mapping(raw_candidate)
            evaluation = self.evaluator.evaluate(candidate)
            artifact_dir = self.workspace / "tool-evaluations" / f"{index:04d}"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            diagnostics = _build_diagnostics(evaluation)
            _write_json(artifact_dir / "candidate.json", candidate.to_dict())
            _write_json(artifact_dir / "diagnostics.json", diagnostics)
            plot_path = _write_plot(artifact_dir, evaluation)
            record = {
                "index": index,
                "candidate": candidate.to_dict(),
                "valid": bool(evaluation.valid),
                "loss": evaluation.loss,
                "message": evaluation.message,
                "diagnostics": str((artifact_dir / "diagnostics.json").relative_to(self.workspace)),
                "plot": (
                    str(plot_path.relative_to(self.workspace)) if plot_path is not None else None
                ),
            }
            self.records.append(record)
            with (self.workspace / "tool-evaluations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return {**record, "used": len(self.records), "maximum": self.max_evaluations}


def _build_diagnostics(evaluation: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid": bool(evaluation.valid),
        "loss": evaluation.loss,
        "message": evaluation.message,
        "crn": str(evaluation.env.state) if evaluation.env is not None else None,
        "trajectories": [],
    }
    outputs = (evaluation.task_info or {}).get("outputs", [])
    for scenario, output in enumerate(outputs):
        array = np.asarray(output, dtype=float)
        if array.size == 0:
            continue
        flattened = array.reshape((-1, array.shape[-1] if array.ndim > 1 else array.size))
        result["trajectories"].append(
            {
                "scenario": scenario,
                "shape": list(array.shape),
                "minimum": float(np.nanmin(array)),
                "maximum": float(np.nanmax(array)),
                "terminal_values": [float(value) for value in flattened[:, -1]],
                "finite": bool(np.isfinite(array).all()),
            }
        )
    return result


def _write_plot(artifact_dir: Path, evaluation: Any) -> Optional[Path]:
    outputs = (evaluation.task_info or {}).get("outputs", [])
    if not outputs:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        line_count = 0
        for scenario, output in enumerate(outputs):
            array = np.asarray(output, dtype=float)
            if array.size == 0:
                continue
            rows = array.reshape((-1, array.shape[-1] if array.ndim > 1 else array.size))
            for output_index, row in enumerate(rows):
                if line_count >= 32:
                    break
                axis.plot(row, alpha=0.7, label=f"scenario {scenario}, output {output_index}")
                line_count += 1
        axis.set_xlabel("time index")
        axis.set_ylabel("output")
        axis.set_title(f"CVODE candidate diagnostics, loss={evaluation.loss}")
        if 0 < line_count <= 12:
            axis.legend(fontsize=7)
        figure.tight_layout()
        path = artifact_dir / "transients.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        return path
    except Exception as exc:
        (artifact_dir / "plot_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
