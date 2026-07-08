"""Reusable LLM graph helpers for CRN proposal workflows.

The objects here keep the graph logic provider-neutral.  A client only needs
``generate_text`` for decision nodes and ``generate_json`` for writer nodes.
Candidate validation and scoring remain delegated to ``LLMCandidateEvaluator``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.memory import LLMMemory
from RL4CRN.llm.prompts import build_candidate_generation_prompt
from RL4CRN.llm.schemas import CandidateEvaluation, LLMCandidate, LLMGenerationConfig, parse_candidates_payload


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


def default_decider_writer_spec() -> LLMGraphSpec:
    """Return a small editable decider -> writer graph."""

    return LLMGraphSpec(
        nodes=[
            LLMGraphNode(
                name="RPA task",
                role="context",
                prompt_template="Task context is provided by the notebook.",
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
                    "Choose a concise CRN proposal strategy for the task below.\n"
                    "Respect these hard constraints:\n"
                    "- exactly {max_added_reactions} reactions;\n"
                    "- use only reaction IDs from the current library;\n"
                    "- no duplicate reaction IDs;\n"
                    "- every parameter must be positive and finite;\n"
                    "- every parameter vector must have the length required by its reaction.\n\n"
                    "Task:\n{task_description}\n\n"
                    "Recent LLM feedback:\n{feedback_text}\n\n"
                    "Best previous LLM candidates:\n{llm_best_text}\n\n"
                    "Forbidden already-evaluated topologies:\n{forbidden_topologies_text}\n\n"
                    "Return a short JSON object with keys strategy and constraints."
                ),
                generation_config=LLMGenerationConfig(temperature=0.2, response_mime_type="text/plain"),
            ),
            LLMGraphNode(
                name="Writer",
                role="json",
                prompt_template=(
                    "{task_description}\n\n"
                    "Decider strategy and constraints:\n{decision}\n\n"
                    "Generate candidates that obey every hard constraint. "
                    "Prefer mechanistically diverse motifs."
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
            ("RPA task", "Decider"),
            ("Search constraints", "Decider"),
            ("Feedback memory", "Decider"),
            ("Decider", "Writer"),
            ("Reaction library", "Writer"),
            ("Writer", "Evaluator"),
            ("Evaluator", "Feedback memory"),
        ],
        title="RPA LLM decider-writer graph",
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
    ):
        self.client = client
        self.evaluator = evaluator
        self.spec = spec or default_decider_writer_spec()
        self.memory = memory or LLMMemory()
        self.comet_logger = comet_logger
        self.metric_prefix = metric_prefix.strip("/")
        self.transcript_jsonl_path = Path(transcript_jsonl_path) if transcript_jsonl_path else None

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
                "RPA task": (-1.6, 0.8),
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

    def decide(self, task_description: str, *, return_prompt: bool = False) -> str | tuple[str, str]:
        """Run the decider node and return its text decision."""
        return self._decide(task_description, forbidden_topologies_text="", return_prompt=return_prompt)

    def _decide(
        self,
        task_description: str,
        *,
        forbidden_topologies_text: str = "",
        return_prompt: bool = False,
    ) -> str | tuple[str, str]:
        """Run the decider node with optional forbidden-topology context."""

        node = self.spec.get_node(self.spec.decider_node)
        prompt = node.prompt_template.format(
            task_description=task_description,
            max_added_reactions=self.evaluator.max_added_reactions,
            feedback_text=self.memory.format_feedback(),
            llm_best_text=self.memory.format_best(),
            forbidden_topologies_text=forbidden_topologies_text or "No forbidden topologies have been archived yet.",
        )

        if hasattr(self.client, "generate_text"):
            decision = str(self.client.generate_text(prompt, generation_config=node.generation_config))
        else:
            payload = self.client.generate_json(prompt, generation_config=node.generation_config)
            decision = json.dumps(payload, indent=2, sort_keys=True)
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
    ) -> str:
        """Build the final JSON-generation prompt for the writer node."""

        node = self.spec.get_node(self.spec.writer_node)
        writer_task = node.prompt_template.format(
            task_description=task_description,
            decision=decision,
            max_added_reactions=self.evaluator.max_added_reactions,
            feedback_text=self.memory.format_feedback(),
            llm_best_text=self.memory.format_best(),
            forbidden_topologies_text=forbidden_topologies_text or "No forbidden topologies have been archived yet.",
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
    ) -> LLMGraphRunResult:
        """Run one graph round, evaluate candidates, update memory, and log."""

        decider_node = self.spec.get_node(self.spec.decider_node)
        decision, decider_prompt = self._decide(
            task_description,
            forbidden_topologies_text=forbidden_topologies_text,
            return_prompt=True,
        )
        writer_node = self.spec.get_node(self.spec.writer_node)
        writer_prompt = self.build_writer_prompt(
            task_description=task_description,
            decision=decision,
            num_candidates=num_candidates,
            hall_of_fame_iter=hall_of_fame_iter,
            forbidden_topologies_text=forbidden_topologies_text,
        )
        messages = [
            self._message_record(
                step=step,
                source_nodes=["RPA task", "Search constraints", "Feedback memory", "Forbidden topology archive"],
                target_node=decider_node.name,
                kind="prompt",
                content=decider_prompt,
                description="Task context, hard constraints, feedback memory, forbidden topology archive, and prior LLM best candidates sent to the Decider.",
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
                source_nodes=[decider_node.name, "Reaction library", "RL Hall of Fame", "Feedback memory", "Forbidden topology archive"],
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
        raw_payload, candidates, evaluations, retry_message = self._generate_evaluate_with_retry(
            writer_prompt=writer_prompt,
            writer_node=writer_node,
            logger=logger or self.comet_logger,
            step=step,
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
        return LLMGraphRunResult(
            spec=self.spec,
            decision=decision,
            writer_prompt=writer_prompt,
            raw_payload=raw_payload,
            candidates=candidates,
            evaluations=evaluations,
        )

    def _generate_evaluate_with_retry(
        self,
        *,
        writer_prompt: str,
        writer_node: LLMGraphNode,
        logger: Any = None,
        step: Optional[int] = None,
    ) -> tuple[Any, List[LLMCandidate], List[CandidateEvaluation], str]:
        """Generate candidates, then retry once with concrete feedback if needed."""

        retry_message = ""
        try:
            raw_payload = self.client.generate_json(
                writer_prompt,
                generation_config=writer_node.generation_config,
            )
            candidates = parse_candidates_payload(raw_payload)
            evaluations = self.evaluator.evaluate_many(candidates)
            if any(evaluation.valid for evaluation in evaluations):
                return raw_payload, candidates, evaluations, retry_message

            retry_message = self._format_evaluation_feedback(evaluations)
            if not retry_message:
                retry_message = "No valid candidates were produced."
        except Exception as exc:
            retry_message = f"{type(exc).__name__}: {exc}"

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
        raw_payload = self.client.generate_json(
            retry_prompt,
            generation_config=writer_node.generation_config,
        )
        candidates = parse_candidates_payload(raw_payload)
        evaluations = self.evaluator.evaluate_many(candidates)
        return raw_payload, candidates, evaluations, retry_message

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
