"""Reusable LLM graph helpers for CRN proposal workflows.

The objects here keep the graph logic provider-neutral.  A client only needs
``generate_text`` for decision nodes and ``generate_json`` for writer nodes.
Candidate validation and scoring remain delegated to ``LLMCandidateEvaluator``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.memory import LLMMemory
from RL4CRN.llm.prompts import build_candidate_generation_prompt, format_hall_of_fame
from RL4CRN.llm.schemas import CandidateEvaluation, LLMCandidate, LLMGenerationConfig, parse_candidates_payload


def _display_rate_bounds(evaluator: Any) -> tuple[float, float]:
    """Return finite prompt bounds, using contract-v2 defaults for generic evaluators."""

    minimum = getattr(evaluator, "min_parameter_value", None)
    maximum = getattr(evaluator, "max_parameter_value", None)
    return (
        0.001 if minimum is None else float(minimum),
        100.0 if maximum is None else float(maximum),
    )


@dataclass(frozen=True)
class LLMGraphNode:
    """One editable role in an LLM proposal graph."""

    name: str
    role: str
    prompt_template: str
    generation_config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)


@dataclass(frozen=True)
class LLMGraphSpec:
    """Declarative graph description for notebooks and reproducible runs."""

    nodes: List[LLMGraphNode]
    edges: List[Tuple[str, str]]
    decider_node: str = "Decider"
    writer_node: str = "Writer"
    title: str = "LLM CRN proposal graph"

    def get_node(self, name: str) -> LLMGraphNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"No LLM graph node named {name!r}.")


@dataclass(frozen=True)
class LLMGraphRunResult:
    """Outputs from one decider-writer proposal round."""

    spec: LLMGraphSpec
    decision: str
    writer_prompt: str
    raw_payload: Any
    candidates: List[LLMCandidate]
    evaluations: List[CandidateEvaluation]
    tool_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    response_validation: Dict[str, Any] = field(default_factory=dict)


def default_decider_writer_spec() -> LLMGraphSpec:
    """Return a small editable decider -> writer graph."""

    return LLMGraphSpec(
        nodes=[
            LLMGraphNode(
                name="Task contract",
                role="context",
                prompt_template="The executable task contract is provided in the workspace.",
            ),
            LLMGraphNode(
                name="Search constraints",
                role="context",
                prompt_template="Budget, library-only reactions, unique IDs, and positive finite parameters.",
            ),
            LLMGraphNode(
                name="Decider",
                role="text",
                prompt_template=(
                    "DECIDER ROLE\n"
                    "Select exactly {num_candidates} concrete candidate CRNs for the task below.\n"
                    "You own every scientific choice: specify each reaction structure and its intended rate. "
                    "You may use equations, species names, reaction-library IDs, tables, or another concise "
                    "notation. Do not emit machine JSON and do not defer design choices to the Writer.\n\n"
                    "The Writer and host will implement these hard constraints:\n"
                    "- exactly {max_added_reactions} reactions;\n"
                    "- only IDs from REACTION_LIBRARY.tsv;\n"
                    "- no duplicate IDs within one candidate;\n"
                    "- one correctly sized parameter vector per reaction;\n"
                    "- finite direct-LLM rates in [{rate_min}, {rate_max}].\n"
                    "A finite intended rate below {rate_min} is truncated to {rate_min}; one above "
                    "{rate_max} is truncated to {rate_max}. Account for that deterministic rule.\n\n"
                    "Task:\n{task_description}\n\n"
                    "Recent LLM feedback:\n{feedback_text}\n\n"
                    "Best previous LLM candidates:\n{llm_best_text}\n\n"
                    "Current ranked RL Hall of Fame:\n{hall_of_fame_text}\n\n"
                    "Latest RL SIL status (optimization context, not candidate quality):\n"
                    "{sil_feedback_text}\n\n"
                    "Forbidden already-evaluated topologies:\n{forbidden_topologies_text}\n\n"
                    "Return a concise, easy-to-read design record containing all {num_candidates} concrete "
                    "CRNs and a short scientific rationale for each. This is not private chain-of-thought."
                ),
                generation_config=LLMGenerationConfig(temperature=0.2, response_mime_type="text/plain"),
            ),
            LLMGraphNode(
                name="Writer",
                role="json",
                prompt_template=(
                    "WRITER ROLE\n"
                    "Implement the Decider's concrete designs as machine JSON. Read "
                    "DECIDER_DESIGNS.md and use targeted lookups in REACTION_LIBRARY.tsv. Preserve the "
                    "Decider's scientific choices; do not invent replacement CRNs or conduct a second "
                    "proposal pass. Resolve structures to allowed IDs, enforce exactly "
                    "{max_added_reactions} unique reactions per member, encode the required parameter "
                    "vectors, and truncate finite rates to [{rate_min}, {rate_max}] before writing JSON.\n\n"
                    "Task contract:\n{task_description}\n\n"
                    "Decider designs:\n{decision}\n\n"
                    "Aim to encode all requested members. Each member will be validated independently, so "
                    "keep every encodable design correct even if another design cannot be represented."
                ),
                generation_config=LLMGenerationConfig(temperature=0.7, max_output_tokens=8192),
            ),
            LLMGraphNode(
                name="Reaction library",
                role="context",
                prompt_template="The writer receives the ID-indexed reaction library.",
            ),
            LLMGraphNode(
                name="Evaluator",
                role="tool",
                prompt_template="Validates JSON candidates and scores them through RL4CRN.",
            ),
            LLMGraphNode(
                name="Feedback memory",
                role="memory",
                prompt_template="Stores recent failures and best LLM-generated candidates.",
            ),
        ],
        edges=[
            ("Task contract", "Decider"),
            ("Search constraints", "Decider"),
            ("Feedback memory", "Decider"),
            ("Decider", "Writer"),
            ("Reaction library", "Writer"),
            ("Writer", "Evaluator"),
            ("Evaluator", "Feedback memory"),
        ],
        title="CRN Decide-then-Write graph",
    )


class DeciderWriterCRNGraph:
    """Run a decider -> writer -> evaluator graph for CRN candidates."""

    def __init__(
        self,
        *,
        client: Any,
        evaluator: LLMCandidateEvaluator,
        spec: Optional[LLMGraphSpec] = None,
        memory: Optional[LLMMemory] = None,
        comet_logger: Any = None,
        metric_prefix: str = "LLM",
        transcript_jsonl_path: Optional[str | Path] = None,
        writer_retry_limit: int = 1,
        workspace_context_mode: bool = False,
    ):
        self.client = client
        self.evaluator = evaluator
        self.spec = spec or default_decider_writer_spec()
        self.memory = memory or LLMMemory()
        self.comet_logger = comet_logger
        self.metric_prefix = metric_prefix.strip("/")
        self.transcript_jsonl_path = Path(transcript_jsonl_path) if transcript_jsonl_path else None
        self.writer_retry_limit = int(writer_retry_limit)
        self.workspace_context_mode = bool(workspace_context_mode)
        if self.writer_retry_limit not in {0, 1}:
            raise ValueError("writer_retry_limit must be 0 or 1.")

    def draw(self, *, ax: Any = None, pos: Optional[Mapping[str, Tuple[float, float]]] = None) -> Any:
        """Draw the graph using networkx and matplotlib, returning the graph."""

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Drawing LLM graphs requires matplotlib and networkx.") from exc

        graph = nx.DiGraph()
        graph.add_nodes_from(node.name for node in self.spec.nodes)
        graph.add_edges_from(self.spec.edges)

        if pos is None:
            pos = {
                "Task contract": (-1.6, 0.8),
                "Search constraints": (-1.6, -0.2),
                "Decider": (0.0, 0.3),
                "Writer": (1.5, 0.3),
                "Reaction library": (1.5, -0.8),
                "Evaluator": (3.0, 0.3),
                "Feedback memory": (1.5, 1.3),
            }

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3.8))
        nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowstyle="-|>", width=1.4, alpha=0.7)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=2500, node_color="#d8e8f5", edgecolors="#355c7d")
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
        ax.set_title(self.spec.title)
        ax.axis("off")
        return graph

    def decide(
        self,
        task_description: str,
        *,
        num_candidates: int = 10,
        return_prompt: bool = False,
    ) -> str | tuple[str, str]:
        """Run the decider node and return its text decision."""
        return self._decide(
            task_description,
            forbidden_topologies_text="",
            hall_of_fame_iter=None,
            sil_feedback_text="",
            num_candidates=num_candidates,
            return_prompt=return_prompt,
        )

    def _decide(
        self,
        task_description: str,
        *,
        forbidden_topologies_text: str = "",
        hall_of_fame_iter: Optional[Iterable[Any]] = None,
        sil_feedback_text: str = "",
        num_candidates: int = 10,
        return_prompt: bool = False,
    ) -> str | tuple[str, str]:
        """Run the decider node with optional forbidden-topology context."""

        node = self.spec.get_node(self.spec.decider_node)
        rate_min, rate_max = _display_rate_bounds(self.evaluator)
        if self.workspace_context_mode:
            feedback_text = "Read prior LLM feedback from CONTEXT/SEARCH_STATE.json."
            llm_best_text = "Read prior best LLM candidates from CONTEXT/SEARCH_STATE.json."
            hall_of_fame_text = "Read the ranked live snapshot from CONTEXT/HALL_OF_FAME.md."
        else:
            feedback_text = self.memory.format_feedback()
            llm_best_text = self.memory.format_best()
            hall_of_fame_text = format_hall_of_fame(hall_of_fame_iter)
        prompt = node.prompt_template.format(
            task_description=task_description,
            num_candidates=int(num_candidates),
            max_added_reactions=self.evaluator.max_added_reactions,
            rate_min=rate_min,
            rate_max=rate_max,
            feedback_text=feedback_text,
            llm_best_text=llm_best_text,
            hall_of_fame_text=hall_of_fame_text,
            sil_feedback_text=sil_feedback_text or "No completed SIL update is available yet.",
            forbidden_topologies_text=forbidden_topologies_text or "No forbidden topologies have been archived yet.",
        )

        if hasattr(self.client, "generate_text"):
            decision = str(self.client.generate_text(prompt, generation_config=node.generation_config))
        else:
            payload = self.client.generate_json(prompt, generation_config=node.generation_config)
            decision = json.dumps(payload, indent=2, sort_keys=True)
        workspace = getattr(self.client, "active_workspace", None)
        if workspace is not None:
            design_path = workspace.path / "DECIDER_DESIGNS.md"
            artifact_decision = (
                design_path.read_text(encoding="utf-8").strip()
                if design_path.is_file()
                else ""
            )
            if artifact_decision:
                if artifact_decision != decision.strip():
                    (workspace.path / "decider_response_reconciliation.json").write_text(
                        json.dumps(
                            {
                                "policy": "workspace-artifact-authoritative",
                                "stdout_matches_artifact": False,
                                "stdout": decision,
                                "artifact": artifact_decision,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                decision = artifact_decision
            else:
                design_path.write_text(decision.rstrip() + "\n", encoding="utf-8")
        if return_prompt:
            return decision, prompt
        return decision

    def build_writer_prompt(
        self,
        *,
        task_description: str,
        decision: str,
        num_candidates: int,
        hall_of_fame_iter: Optional[Iterable[Any]] = None,
        forbidden_topologies_text: str = "",
        sil_feedback_text: str = "",
    ) -> str:
        """Build the final JSON-generation prompt for the writer node."""

        node = self.spec.get_node(self.spec.writer_node)
        rate_min, rate_max = _display_rate_bounds(self.evaluator)
        writer_decision = (
            "Read and implement the exact design record in DECIDER_DESIGNS.md."
            if self.workspace_context_mode
            else decision
        )
        writer_task = node.prompt_template.format(
            task_description=task_description,
            decision=writer_decision,
            num_candidates=int(num_candidates),
            max_added_reactions=self.evaluator.max_added_reactions,
            rate_min=rate_min,
            rate_max=rate_max,
            feedback_text=self.memory.format_feedback(),
            llm_best_text=self.memory.format_best(),
            hall_of_fame_text=format_hall_of_fame(hall_of_fame_iter),
            sil_feedback_text=sil_feedback_text or "No completed SIL update is available yet.",
            forbidden_topologies_text=forbidden_topologies_text or "No forbidden topologies have been archived yet.",
        )
        if self.workspace_context_mode:
            return writer_task + (
                "\n\nEncoding resources are workspace files: use OUTPUT_GUIDE.json for the "
                "machine schema, REACTION_LIBRARY.tsv for allowed IDs and arities, and "
                "CONTEXT/ for live search state. Do not reproduce those files in the response."
            )
        return build_candidate_generation_prompt(
            task_description=writer_task,
            reaction_library=self.evaluator.library,
            max_added_reactions=self.evaluator.max_added_reactions,
            num_candidates=num_candidates,
            hall_of_fame_iter=hall_of_fame_iter,
            feedback_text=self.memory.format_feedback(),
            llm_best_text=self.memory.format_best(),
            forbidden_topologies_text=forbidden_topologies_text,
            sil_feedback_text=sil_feedback_text,
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
    ) -> LLMGraphRunResult:
        """Run one graph round, evaluate candidates, update memory, and log."""

        round_tic = time.perf_counter()
        decider_node = self.spec.get_node(self.spec.decider_node)
        decider_tic = time.perf_counter()
        decision, decider_prompt = self._decide(
            task_description,
            forbidden_topologies_text=forbidden_topologies_text,
            hall_of_fame_iter=hall_of_fame_iter,
            sil_feedback_text=sil_feedback_text,
            num_candidates=num_candidates,
            return_prompt=True,
        )
        decider_seconds = time.perf_counter() - decider_tic
        writer_node = self.spec.get_node(self.spec.writer_node)
        writer_prompt = self.build_writer_prompt(
            task_description=task_description,
            decision=decision,
            num_candidates=num_candidates,
            hall_of_fame_iter=hall_of_fame_iter,
            forbidden_topologies_text=forbidden_topologies_text,
            sil_feedback_text=sil_feedback_text,
        )
        messages = [
            self._message_record(
                step=step,
                source_nodes=["Task contract", "Search constraints", "RL Hall of Fame", "RL SIL status", "Feedback memory", "Forbidden topology archive"],
                target_node=decider_node.name,
                kind="prompt",
                content=decider_prompt,
                description="Task context, hard constraints, ranked Hall of Fame, SIL status, feedback memory, forbidden topology archive, and prior LLM best candidates sent to the Decider.",
            ),
            self._message_record(
                step=step,
                source_nodes=[decider_node.name],
                target_node=writer_node.name,
                kind="response",
                content=decision,
                description="Decider strategy returned to the Writer.",
            ),
            self._message_record(
                step=step,
                source_nodes=[decider_node.name, "Reaction library", "RL Hall of Fame", "RL SIL status", "Feedback memory", "Forbidden topology archive"],
                target_node=writer_node.name,
                kind="prompt",
                content=writer_prompt,
                description="Writer prompt containing the decider strategy, reaction library, current Hall of Fame, forbidden topology archive, and output schema.",
            ),
        ]
        self.log_messages(
            messages=messages,
            logger=logger or self.comet_logger,
            step=step,
            name="discussion",
        )
        writer_eval_tic = time.perf_counter()
        raw_payload, candidates, evaluations, retry_message, timing = self._generate_evaluate_with_retry(
            writer_prompt=writer_prompt,
            writer_node=writer_node,
            logger=logger or self.comet_logger,
            step=step,
        )
        writer_and_evaluation_seconds = time.perf_counter() - writer_eval_tic
        response_validation = dict(
            getattr(self.client, "last_response_validation", {}) or {}
        )
        for i, evaluation in enumerate(evaluations):
            if evaluation.env is not None:
                is_forbidden = (
                    hasattr(self.evaluator, "is_forbidden_env")
                    and self.evaluator.is_forbidden_env(evaluation.env)
                )
                info = dict(getattr(evaluation.env.state, "last_task_info", {}) or {})
                info.update(
                    {
                        "source": "LLM",
                        "llm_graph": self.spec.title,
                        "llm_candidate_index": i,
                        "forbidden_topology": bool(info.get("forbidden_topology", False) or is_forbidden),
                    }
                )
                evaluation.env.state.last_task_info = info
                object.__setattr__(evaluation, "task_info", info)
                if is_forbidden:
                    object.__setattr__(evaluation, "valid", False)
                    object.__setattr__(
                        evaluation,
                        "message",
                        "forbidden topology: blocked before Hall-of-Fame insertion.",
                    )
                if add_to_hall_of_fame is not None and evaluation.valid and not is_forbidden:
                    add_to_hall_of_fame.add(evaluation.env)
        if jsonl_path is not None:
            self.evaluator.append_jsonl(evaluations, jsonl_path)
        self.memory.update_many(evaluations)
        self.log_round(evaluations, logger=logger or self.comet_logger, step=step)
        payload_messages = [
            self._message_record(
                step=step,
                source_nodes=[writer_node.name],
                target_node="Evaluator",
                kind="json_payload",
                content=raw_payload,
                description="Writer JSON payload parsed into candidate CRNs.",
            ),
            self._message_record(
                step=step,
                source_nodes=["Evaluator"],
                target_node="Feedback memory",
                kind="evaluation_summary",
                content=[ev.to_log_record(include_crn=False) for ev in evaluations],
                description="Evaluator messages, losses, and validity flags passed back to memory and the shared Hall of Fame.",
            ),
        ]
        self.log_messages(
            messages=payload_messages,
            logger=logger or self.comet_logger,
            step=step,
            name="payload_and_evaluation",
        )
        self.log_payload(
            raw_payload=raw_payload,
            evaluations=evaluations,
            retry_message=retry_message,
            logger=logger or self.comet_logger,
            step=step,
        )
        total_seconds = time.perf_counter() - round_tic
        self.log_timing(
            {
                "Round Seconds": total_seconds,
                "Decider Seconds": decider_seconds,
                "Writer And Evaluation Seconds": writer_and_evaluation_seconds,
                **timing,
            },
            logger=logger or self.comet_logger,
            step=step,
        )
        self.log_response_validation(
            response_validation,
            logger=logger or self.comet_logger,
            step=step,
        )
        return LLMGraphRunResult(
            spec=self.spec,
            decision=decision,
            writer_prompt=writer_prompt,
            raw_payload=raw_payload,
            candidates=candidates,
            evaluations=evaluations,
            response_validation=response_validation,
        )

    def _generate_evaluate_with_retry(
        self,
        *,
        writer_prompt: str,
        writer_node: LLMGraphNode,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> tuple[Any, List[LLMCandidate], List[CandidateEvaluation], str, Dict[str, float]]:
        """Generate candidates, then retry once with concrete feedback if needed."""

        retry_message = ""
        timing: Dict[str, float] = {
            "Writer Generate Seconds": 0.0,
            "Candidate Evaluation Seconds": 0.0,
            "Retry Writer Generate Seconds": 0.0,
            "Retry Candidate Evaluation Seconds": 0.0,
        }
        try:
            tic = time.perf_counter()
            raw_payload = self.client.generate_json(
                writer_prompt,
                generation_config=writer_node.generation_config,
            )
            timing["Writer Generate Seconds"] = time.perf_counter() - tic
            candidates = parse_candidates_payload(raw_payload)
            tic = time.perf_counter()
            evaluations = self.evaluator.evaluate_many(candidates)
            timing["Candidate Evaluation Seconds"] = time.perf_counter() - tic
            if any(evaluation.valid for evaluation in evaluations):
                return raw_payload, candidates, evaluations, retry_message, timing

            retry_message = self._format_evaluation_feedback(evaluations)
            if not retry_message:
                retry_message = "No valid candidates were produced."
            if self.writer_retry_limit == 0:
                return raw_payload, candidates, evaluations, retry_message, timing
        except Exception as exc:
            retry_message = f"{type(exc).__name__}: {exc}"
            if self.writer_retry_limit == 0:
                self.log_response_validation(
                    dict(getattr(self.client, "last_response_validation", {}) or {}),
                    logger=logger,
                    step=step,
                )
                raise

        retry_prompt = self._build_retry_prompt(writer_prompt, retry_message)
        retry_messages = [
            self._message_record(
                step=step,
                source_nodes=["Evaluator"],
                target_node=writer_node.name,
                kind="retry_feedback",
                content=retry_message,
                description="Concrete parsing or validation error fed back to the Writer for one repair attempt.",
            ),
            self._message_record(
                step=step,
                source_nodes=["Evaluator", "Feedback memory"],
                target_node=writer_node.name,
                kind="retry_prompt",
                content=retry_prompt,
                description="Full retry prompt sent to the Writer.",
            ),
        ]
        self.log_messages(
            messages=retry_messages,
            logger=logger,
            step=step,
            name="retry",
        )
        self.log_retry(
            retry_message=retry_message,
            retry_prompt=retry_prompt,
            logger=logger,
            step=step,
        )
        tic = time.perf_counter()
        raw_payload = self.client.generate_json(
            retry_prompt,
            generation_config=writer_node.generation_config,
        )
        timing["Retry Writer Generate Seconds"] = time.perf_counter() - tic
        candidates = parse_candidates_payload(raw_payload)
        tic = time.perf_counter()
        evaluations = self.evaluator.evaluate_many(candidates)
        timing["Retry Candidate Evaluation Seconds"] = time.perf_counter() - tic
        return raw_payload, candidates, evaluations, retry_message, timing

    @staticmethod
    def _build_retry_prompt(writer_prompt: str, retry_message: str) -> str:
        return (
            f"{writer_prompt}\n\n"
            "=== Previous Attempt Failed ===\n"
            f"{retry_message}\n\n"
            "Repair the answer now. Return only valid JSON with this exact shape:\n"
            "{\n"
            '  "candidates": [\n'
            "    {\n"
            '      "reaction_ids": [0, 1],\n'
            '      "parameter_values": [[1.0], [0.5, 2.0]]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not include reasoning strings, markdown, or extra keys."
        )

    @staticmethod
    def _format_evaluation_feedback(evaluations: Sequence[CandidateEvaluation], limit: int = 5) -> str:
        messages = []
        for i, evaluation in enumerate(evaluations[: int(limit)]):
            messages.append(f"candidate {i}: {evaluation.message}")
        return "\n".join(messages)

    def log_discussion(
        self,
        *,
        decision: str,
        writer_prompt: str,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log the decider/writer discussion in a human-readable Comet panel."""

        logger = logger or self.comet_logger
        if logger is None:
            return

        transcript = (
            f"LLM graph: {self.spec.title}\n"
            f"Step: {step}\n\n"
            "=== Decider reasoning ===\n"
            f"{decision}\n\n"
            "=== Writer prompt ===\n"
            f"{writer_prompt}\n"
        )
        if hasattr(logger, "log_text"):
            logger.log_text(transcript)
        if hasattr(logger, "log_asset_data"):
            logger.log_asset_data(
                transcript,
                name=f"llm_discussion_step_{step if step is not None else 'na'}.txt",
                step=step,
            )

    def log_messages(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        logger: Any = None,
        step: Optional[int] = None,
        name: str = "messages",
    ) -> None:
        """Persist graph messages locally and to Comet with explicit node flow."""

        if not messages:
            return
        self._append_transcript_jsonl(messages)

        logger = logger or self.comet_logger
        if logger is None:
            return

        text = self._format_message_transcript(messages)
        if hasattr(logger, "log_text"):
            logger.log_text(text)
        if hasattr(logger, "log_asset_data"):
            safe_step = step if step is not None else "na"
            logger.log_asset_data(
                text,
                name=f"llm_{name}_step_{safe_step}.txt",
                step=step,
            )
            logger.log_asset_data(
                json.dumps({"messages": messages}, indent=2, sort_keys=True),
                name=f"llm_{name}_step_{safe_step}.json",
                step=step,
            )

    def _append_transcript_jsonl(self, messages: Sequence[Dict[str, Any]]) -> None:
        if self.transcript_jsonl_path is None:
            return
        self.transcript_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript_jsonl_path.open("a", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message, sort_keys=True) + "\n")

    @staticmethod
    def _message_record(
        *,
        step: Optional[int],
        source_nodes: Sequence[str],
        target_node: str,
        kind: str,
        content: Any,
        description: str,
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "source_nodes": list(source_nodes),
            "target_node": target_node,
            "edge": f"{' + '.join(source_nodes)} -> {target_node}",
            "kind": kind,
            "description": description,
            "content": content,
        }

    @staticmethod
    def _format_message_transcript(messages: Sequence[Dict[str, Any]]) -> str:
        blocks = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, indent=2, sort_keys=True)
            blocks.append(
                f"[step {message.get('step')}] {message.get('edge')} | {message.get('kind')}\n"
                f"{message.get('description')}\n\n"
                f"{content}"
            )
        return "\n\n---\n\n".join(blocks)

    def log_payload(
        self,
        *,
        raw_payload: Any,
        evaluations: Sequence[CandidateEvaluation],
        retry_message: str = "",
        logger: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log the structured writer output and evaluation summary."""

        logger = logger or self.comet_logger
        if logger is None or not hasattr(logger, "log_asset_data"):
            return

        record = {
            "graph": self.spec.title,
            "step": step,
            "retry_message": retry_message,
            "raw_payload": raw_payload,
            "evaluations": [ev.to_log_record(include_crn=True) for ev in evaluations],
        }
        logger.log_asset_data(
            json.dumps(record, indent=2, sort_keys=True),
            name=f"llm_candidates_step_{step if step is not None else 'na'}.json",
            step=step,
        )

    def log_retry(
        self,
        *,
        retry_message: str,
        retry_prompt: str,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log the feedback used for the single writer retry."""

        logger = logger or self.comet_logger
        if logger is None:
            return
        transcript = (
            f"LLM graph: {self.spec.title}\n"
            f"Step: {step}\n\n"
            "=== Writer retry feedback ===\n"
            f"{retry_message}\n\n"
            "=== Retry prompt ===\n"
            f"{retry_prompt}\n"
        )
        if hasattr(logger, "log_text"):
            logger.log_text(transcript)
        if hasattr(logger, "log_asset_data"):
            logger.log_asset_data(
                transcript,
                name=f"llm_retry_step_{step if step is not None else 'na'}.txt",
                step=step,
            )

    def log_round(
        self,
        evaluations: Sequence[CandidateEvaluation],
        *,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log LLM metrics with names separated from RL metrics."""

        logger = logger or self.comet_logger
        if logger is None or not hasattr(logger, "log_metric"):
            return

        losses = [float(ev.loss) for ev in evaluations if ev.valid and ev.loss is not None]
        valid_count = sum(1 for ev in evaluations if ev.valid)
        prefix = self.metric_prefix
        logger.log_metric(f"{prefix}/Valid Count", valid_count, step=step)
        logger.log_metric(f"{prefix}/Invalid Count", len(evaluations) - valid_count, step=step)
        if losses:
            logger.log_metric(f"{prefix}/Loss Best", min(losses), step=step)
            logger.log_metric(f"{prefix}/Loss Average", sum(losses) / len(losses), step=step)
            logger.log_metric(f"{prefix}/Loss Worst", max(losses), step=step)
            for i, loss in enumerate(losses):
                logger.log_metric(f"{prefix}/Loss Candidate {i}", loss, step=step)
                series_step = i if step is None else int(step) * 1000 + i
                logger.log_metric(f"{prefix}/Loss Candidate Series", loss, step=series_step)

    def log_timing(
        self,
        timing: Mapping[str, float],
        *,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log graph timing metrics with the LLM prefix."""

        logger = logger or self.comet_logger
        if logger is None or not hasattr(logger, "log_metric"):
            return
        prefix = self.metric_prefix
        for name, value in timing.items():
            logger.log_metric(f"{prefix}/Timing {name}", float(value), step=step)

    def log_response_validation(
        self,
        validation: Mapping[str, Any],
        *,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Mirror member-level Writer validation and rate clamping to Comet."""

        if not validation:
            return
        logger = logger or self.comet_logger
        if logger is None:
            return
        prefix = self.metric_prefix
        if hasattr(logger, "log_metric"):
            for field, label in (
                ("returned_candidate_count", "Writer Returned Count"),
                ("accepted_candidate_count", "Writer Accepted Count"),
                ("clamped_parameter_count", "Writer Clamped Parameter Count"),
            ):
                if validation.get(field) is not None:
                    logger.log_metric(
                        f"{prefix}/{label}", int(validation[field]), step=step
                    )
            logger.log_metric(
                f"{prefix}/Writer Rejected Count",
                len(validation.get("rejected_candidates", ()) or ()),
                step=step,
            )
        if hasattr(logger, "log_asset_data"):
            logger.log_asset_data(
                json.dumps(dict(validation), indent=2, sort_keys=True),
                name=f"llm_member_validation_step_{step if step is not None else 'na'}.json",
                step=step,
            )


def plot_llm_evaluations(
    evaluations: Sequence[CandidateEvaluation],
    *,
    ax: Any = None,
    title: str = "LLM candidate losses",
) -> Any:
    """Plot valid LLM candidate losses with a label distinct from RL metrics."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting LLM evaluations requires matplotlib.") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    xs = []
    ys = []
    for i, evaluation in enumerate(evaluations):
        if evaluation.valid and evaluation.loss is not None:
            xs.append(i)
            ys.append(float(evaluation.loss))
    ax.plot(xs, ys, marker="o", color="#2f6f9f", label="LLM candidates")
    ax.set_xlabel("LLM candidate index")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return ax
