"""CRN generators that persist complete DeepSeek Harness run artifacts."""

from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
import json
from io import BytesIO
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from RL4CRN.llm.generator import LLMCRNGenerator, LLMGenerationRound
from RL4CRN.llm.graphs import DeciderWriterCRNGraph, LLMGraphRunResult
from RL4CRN.llm.harness_client import HarnessLLMClient, build_crn_output_contract
from RL4CRN.llm.schemas import parse_candidates_payload
from RL4CRN.llm.workspace_tools import (
    WorkspaceEvaluationService,
    WorkspaceLiteratureService,
    default_workspace_tool_files,
)
from RL4CRN.utils.results_database import serialize_crn


class HarnessCRNGenerator(LLMCRNGenerator):
    """Single-agent CRN generator backed by an on-demand Harness process."""

    client: HarnessLLMClient

    def __init__(
        self,
        *args: Any,
        max_workspace_evaluations: int = 10,
        literature_database: Optional[str | Path] = None,
        max_literature_searches: int = 2,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.max_workspace_evaluations = int(max_workspace_evaluations)
        self.literature_database = Path(literature_database) if literature_database else None
        self.max_literature_searches = int(max_literature_searches)

    def fork(self) -> "HarnessCRNGenerator":
        """Create isolated mutable client and memory state for a concurrent round."""

        return type(self)(
            client=self.client.fork(),
            evaluator=self.evaluator,
            memory=deepcopy(self.memory),
            system_prompt=self.system_prompt,
            generation_config=self.generation_config,
            max_workspace_evaluations=self.max_workspace_evaluations,
            literature_database=self.literature_database,
            max_literature_searches=self.max_literature_searches,
        )

    def run_round(
        self,
        *,
        task_description: str,
        num_candidates: int = 10,
        hall_of_fame_iter: Optional[Iterable[Any]] = None,
        add_to_hall_of_fame: Optional[Any] = None,
        jsonl_path: Optional[str | Path] = None,
        logger: Any = None,
        step: Optional[int] = None,
        forbidden_topologies_text: str = "",
        sil_feedback_text: str = "",
    ) -> LLMGenerationRound:
        contract = build_crn_output_contract(
            self.evaluator,
            num_candidates=num_candidates,
            task_description=task_description,
            candidate_validation_policy=self.client.candidate_validation_policy,
        )
        context_kwargs = {
            "hall_of_fame_iter": hall_of_fame_iter,
            "forbidden_topologies_text": forbidden_topologies_text,
            "sil_feedback_text": sil_feedback_text,
            "step": step,
        }
        workspace_files = default_workspace_tool_files(
            include_literature=self.literature_database is not None
        )
        workspace_files.update(_workspace_context_files(context_kwargs, self.memory))
        with self.client.run(
            task_description=task_description,
            contract=contract,
            workspace_files=workspace_files,
            label="crn-generation",
        ) as workspace:
            prompt = _single_request_prompt(num_candidates)
            with ExitStack() as stack:
                tool_service = stack.enter_context(
                    WorkspaceEvaluationService(
                        workspace.path,
                        self.evaluator,
                        max_evaluations=self.max_workspace_evaluations,
                    )
                )
                literature_service = None
                if self.literature_database is not None:
                    literature_service = stack.enter_context(
                        WorkspaceLiteratureService(
                            workspace.path,
                            self.literature_database,
                            max_searches=self.max_literature_searches,
                        )
                    )
                raw_payload = self.client.generate_json(
                    prompt,
                    generation_config=self.generation_config,
                )
            candidates = parse_candidates_payload(raw_payload)
            evaluations = self.evaluator.evaluate_many(
                candidates,
                add_to_hall_of_fame=add_to_hall_of_fame,
                jsonl_path=jsonl_path,
            )
            self.memory.update_many(evaluations)
            result = LLMGenerationRound(
                prompt=prompt,
                candidates=candidates,
                evaluations=evaluations,
                raw_payload=raw_payload,
                tool_evaluations=list(tool_service.records),
            )
            _persist_round(workspace.path, result.raw_payload, result.candidates, result.evaluations)
            workspace.write_evaluations(result.evaluations)
            _append_single_request_notes(workspace.path, result)
            active_logger = logger
            if active_logger is not None and hasattr(active_logger, "log_metric"):
                active_logger.log_metric("LLM/Model Requests", 1, step=step)
                active_logger.log_metric(
                    "LLM/Workspace Tool Evaluations", len(result.tool_evaluations), step=step
                )
                if literature_service is not None:
                    active_logger.log_metric(
                        "LLM/Literature Searches", len(literature_service.records), step=step
                    )
            return result


class HarnessDeciderWriterCRNGraph(DeciderWriterCRNGraph):
    """Decider/writer graph with one isolated workspace per graph round."""

    client: HarnessLLMClient

    def __init__(
        self,
        *args: Any,
        max_workspace_evaluations: int = 10,
        literature_database: Optional[str | Path] = None,
        max_literature_searches: int = 2,
        **kwargs: Any,
    ):
        kwargs.setdefault("workspace_context_mode", True)
        super().__init__(*args, **kwargs)
        self.max_workspace_evaluations = int(max_workspace_evaluations)
        self.literature_database = Path(literature_database) if literature_database else None
        self.max_literature_searches = int(max_literature_searches)

    def fork(self) -> "HarnessDeciderWriterCRNGraph":
        """Create isolated client and memory state for a concurrent graph round."""

        return type(self)(
            client=self.client.fork(),
            evaluator=self.evaluator,
            spec=self.spec,
            memory=deepcopy(self.memory),
            comet_logger=self.comet_logger,
            metric_prefix=self.metric_prefix,
            transcript_jsonl_path=self.transcript_jsonl_path,
            writer_retry_limit=self.writer_retry_limit,
            workspace_context_mode=self.workspace_context_mode,
            max_workspace_evaluations=self.max_workspace_evaluations,
            literature_database=self.literature_database,
            max_literature_searches=self.max_literature_searches,
        )

    def run_round(self, *, task_description: str, num_candidates: int = 10, **kwargs: Any) -> LLMGraphRunResult:
        contract = build_crn_output_contract(
            self.evaluator,
            num_candidates=num_candidates,
            task_description=task_description,
            candidate_validation_policy=self.client.candidate_validation_policy,
        )
        workspace_files = default_workspace_tool_files(
            include_literature=self.literature_database is not None
        )
        workspace_files.update(_workspace_context_files(kwargs, self.memory))
        with self.client.run(
            task_description=task_description,
            contract=contract,
            workspace_files=workspace_files,
            label="crn-decider-writer",
        ) as workspace:
            graph_kwargs = dict(kwargs)
            graph_kwargs["hall_of_fame_iter"] = None
            graph_kwargs["forbidden_topologies_text"] = (
                "Read CONTEXT/EXCLUDED_TOPOLOGIES.md; do not reproduce its contents in prompts."
            )
            graph_kwargs["sil_feedback_text"] = (
                "Read CONTEXT/SIL_STATUS.md; do not reproduce its contents in prompts."
            )
            with ExitStack() as stack:
                tool_service = stack.enter_context(
                    WorkspaceEvaluationService(
                        workspace.path,
                        self.evaluator,
                        max_evaluations=self.max_workspace_evaluations,
                    )
                )
                literature_service = None
                if self.literature_database is not None:
                    literature_service = stack.enter_context(
                        WorkspaceLiteratureService(
                            workspace.path,
                            self.literature_database,
                            max_searches=self.max_literature_searches,
                        )
                    )
                result = super().run_round(
                    task_description=task_description,
                    num_candidates=num_candidates,
                    **graph_kwargs,
                )
            result = replace(
                result,
                tool_evaluations=list(tool_service.records),
                response_validation={
                    **result.response_validation,
                    "provider_call_count": int(workspace.call_count),
                },
            )
            _persist_round(workspace.path, result.raw_payload, result.candidates, result.evaluations)
            (workspace.path / "decision.txt").write_text(result.decision.rstrip() + "\n", encoding="utf-8")
            (workspace.path / "DECIDER_DESIGNS.md").write_text(
                result.decision.rstrip() + "\n", encoding="utf-8"
            )
            (workspace.path / "writer_prompt.txt").write_text(
                result.writer_prompt.rstrip() + "\n", encoding="utf-8"
            )
            (workspace.path / "WRITER_PAYLOAD.json").write_text(
                json.dumps(result.raw_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (workspace.path / "response_validation_summary.json").write_text(
                json.dumps(result.response_validation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            workspace.write_evaluations(result.evaluations)
            _append_evaluator_notes(workspace.path, result)
            logger = kwargs.get("logger") or self.comet_logger
            step = kwargs.get("step")
            if logger is not None and hasattr(logger, "log_metric"):
                logger.log_metric("LLM/Model Requests", workspace.call_count, step=step)
                logger.log_metric(
                    "LLM/Decider Requests", min(workspace.call_count, 1), step=step
                )
                logger.log_metric(
                    "LLM/Writer Requests",
                    max(0, min(workspace.call_count - 1, 1)),
                    step=step,
                )
                logger.log_metric(
                    "LLM/Workspace Tool Evaluations", len(result.tool_evaluations), step=step
                )
                if literature_service is not None:
                    logger.log_metric(
                        "LLM/Literature Searches", len(literature_service.records), step=step
                    )
            return result


def _persist_round(path: Path, raw_payload: Any, candidates: Any, evaluations: Any) -> None:
    (path / "raw_payload.json").write_text(
        json.dumps(raw_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "candidates.json").write_text(
        json.dumps(
            {"candidates": [candidate.to_dict() for candidate in candidates]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _single_request_prompt(num_candidates: int) -> str:
    novel_count = max(1, (int(num_candidates) * 3 + 4) // 5)
    refinement_count = int(num_candidates) - novel_count
    return (
        f"Generate exactly {int(num_candidates)} distinct CRN candidates in one response. "
        "Use TASK.md for the benchmark, OUTPUT_GUIDE.json for the machine schema and "
        "constraints, targeted grep queries on REACTION_LIBRARY.tsv for reaction IDs, "
        "and CONTEXT/ for the live Hall of Fame, SIL status, "
        "excluded topologies, and cached diagnostics. "
        f"Propose {novel_count} new reaction-ID sets and {refinement_count} parameter "
        "refinements of promising admissible Hall-of-Fame sets. You may inspect cached "
        "evidence. Prefer proposing now and using the external results in the next scheduled "
        "call. Only if one concrete uncertainty could materially alter this batch, use at most "
        "one batched workspace evaluator request with up to three carefully selected probes. Update "
        "REASONING_NOTES.md with concise scientific rationale, then write the exact answer "
        "to FINAL_RESPONSE.json. Return the identical JSON object "
        "matching OUTPUT_GUIDE.json; do not return commentary or make another proposal pass."
    )


def _append_single_request_notes(path: Path, result: LLMGenerationRound) -> None:
    valid = [evaluation for evaluation in result.evaluations if evaluation.valid]
    best = min((evaluation.loss for evaluation in valid if evaluation.loss is not None), default=None)
    with (path / "REASONING_NOTES.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## External evaluator outcome\n\n"
            "- Model requests for this proposal batch: 1\n"
            f"- Final candidates evaluated: {len(result.evaluations)}\n"
            f"- Valid final candidates: {len(valid)}\n"
            f"- Best final loss: {best}\n"
            f"- Exploratory tool evaluations used: {len(result.tool_evaluations)}\n"
        )


def _workspace_context_files(kwargs: Mapping[str, Any], memory: Any) -> Dict[str, Any]:
    hall_envs = list(kwargs.get("hall_of_fame_iter") or ())
    hall = [_hall_entry(rank, env) for rank, env in enumerate(hall_envs)]
    files: Dict[str, Any] = {
        "CONTEXT/HALL_OF_FAME.json": {"entries": hall},
        "CONTEXT/HALL_OF_FAME.md": _hall_markdown(hall),
        "CONTEXT/SIL_STATUS.md": kwargs.get("sil_feedback_text")
        or "No completed SIL update is available yet.",
        "CONTEXT/EXCLUDED_TOPOLOGIES.md": kwargs.get("forbidden_topologies_text")
        or "No fully processed topologies have been excluded yet.",
        "CONTEXT/SEARCH_STATE.json": {
            "rl_epoch_at_snapshot": kwargs.get("step"),
            "hall_of_fame_size": len(hall),
            "llm_feedback": memory.format_feedback(),
            "llm_best": memory.format_best(),
            "dynamic_context_location": "CONTEXT/",
        },
    }
    for rank, env in enumerate(hall_envs):
        info = dict(getattr(env.state, "last_task_info", {}) or {})
        diagnostics = _cached_trajectory_diagnostics(info.get("outputs", ()))
        artifact_root = f"CONTEXT/hall-of-fame/rank-{rank:03d}"
        files[f"{artifact_root}/diagnostics.json"] = {
            "source": "cached_live_run",
            "loss": info.get("reward"),
            "trajectories": diagnostics,
        }
        plot = _cached_trajectory_plot(info.get("outputs", ()), loss=info.get("reward"))
        if plot is not None:
            files[f"{artifact_root}/transients.jpg"] = plot
    return files


def _hall_entry(rank: int, env: Any) -> Dict[str, Any]:
    state = env.state
    serialized = serialize_crn(state)
    info = dict(getattr(state, "last_task_info", {}) or {})
    actions = []
    for action in getattr(state, "raw_actions_taken", ()) or ():
        actions.append(
            {
                "reaction_id": action.get("reaction index"),
                "parameter_values": action.get("parameters", action.get("continuous parameters")),
            }
        )
    return {
        "rank": rank,
        "topology_hash": serialized["topology_hash"],
        "loss": info.get("reward"),
        "actions": actions,
        "crn": str(state),
        "source": info.get("source", "RL"),
        "cached_diagnostics": f"CONTEXT/hall-of-fame/rank-{rank:03d}/diagnostics.json",
        "cached_plot": f"CONTEXT/hall-of-fame/rank-{rank:03d}/transients.jpg",
    }


def _hall_markdown(entries: Iterable[Mapping[str, Any]]) -> str:
    entries = list(entries)
    if not entries:
        return "# Hall of Fame\n\nNo Hall-of-Fame entries are available at this snapshot."
    lines = ["# Hall of Fame", "", "Ranked by lower loss. Use entries for refinement or mechanistic comparison.", ""]
    for entry in entries:
        lines.extend(
            [
                f"## Rank {entry['rank']}",
                f"- Loss: {entry.get('loss')}",
                f"- Source: {entry.get('source')}",
                f"- Actions: `{json.dumps(entry.get('actions'), sort_keys=True)}`",
                f"- CRN: `{entry.get('crn')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _cached_trajectory_diagnostics(outputs: Iterable[Any]) -> list[Dict[str, Any]]:
    diagnostics = []
    for scenario, output in enumerate(outputs):
        array = np.asarray(output, dtype=float)
        if array.size == 0:
            continue
        rows = array.reshape((-1, array.shape[-1] if array.ndim > 1 else array.size))
        diagnostics.append(
            {
                "scenario": scenario,
                "shape": list(array.shape),
                "minimum": float(np.nanmin(array)),
                "maximum": float(np.nanmax(array)),
                "terminal_values": [float(value) for value in rows[:, -1]],
                "finite": bool(np.isfinite(array).all()),
            }
        )
    return diagnostics


def _cached_trajectory_plot(outputs: Iterable[Any], *, loss: Any) -> Optional[bytes]:
    outputs = list(outputs)
    if not outputs:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        lines = 0
        for scenario, output in enumerate(outputs):
            array = np.asarray(output, dtype=float)
            if array.size == 0:
                continue
            rows = array.reshape((-1, array.shape[-1] if array.ndim > 1 else array.size))
            for output_index, row in enumerate(rows):
                if lines >= 32:
                    break
                axis.plot(row, alpha=0.7, label=f"scenario {scenario}, output {output_index}")
                lines += 1
        axis.set_xlabel("time index")
        axis.set_ylabel("output")
        axis.set_title(f"Cached live-run trajectories, loss={loss}")
        if 0 < lines <= 12:
            axis.legend(fontsize=7)
        figure.tight_layout()
        buffer = BytesIO()
        figure.savefig(buffer, format="jpeg", dpi=140, facecolor="white")
        plt.close(figure)
        return buffer.getvalue()
    except Exception:
        return None


def _append_evaluator_notes(path: Path, result: LLMGraphRunResult) -> None:
    valid = [evaluation for evaluation in result.evaluations if evaluation.valid]
    best = min((evaluation.loss for evaluation in valid if evaluation.loss is not None), default=None)
    with (path / "REASONING_NOTES.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Decider design summary\n\n"
            f"{result.decision.strip()[:4000]}\n"
            "\n## Writer and validation outcome\n\n"
            "- Provider calls: 2 (Decider, then Writer)\n"
            f"- Writer members returned: {result.response_validation.get('returned_candidate_count')}\n"
            f"- Writer members accepted structurally: {result.response_validation.get('accepted_candidate_count')}\n"
            f"- Writer members rejected structurally: {len(result.response_validation.get('rejected_candidates', ()) or ())}\n"
            f"- Rate values clamped by host: {result.response_validation.get('clamped_parameter_count', 0)}\n"
            "\n## External evaluator outcome\n\n"
            f"- Final candidates evaluated: {len(result.evaluations)}\n"
            f"- Valid final candidates: {len(valid)}\n"
            f"- Best final loss: {best}\n"
            f"- Exploratory tool evaluations used: {len(result.tool_evaluations)}\n"
        )
