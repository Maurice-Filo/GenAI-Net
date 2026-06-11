"""Small memory buffers for LLM-assisted CRN search."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional

from RL4CRN.llm.schemas import CandidateEvaluation


@dataclass(frozen=True)
class LLMFeedbackRecord:
    """A recent LLM proposal and its evaluation outcome."""

    reasoning: str
    valid: bool
    loss: Optional[float]
    message: str = ""

    def to_prompt_text(self) -> str:
        status = "valid" if self.valid else "invalid"
        loss = "N/A" if self.loss is None else f"{self.loss:.6g}"
        return f"- {status}; loss={loss}; note={self.message}; reasoning={self.reasoning}"


class LLMMemory:
    """Keep recent feedback and the best evaluated LLM candidates.

    The Hall of Fame used by the RL trainer stores full environments.  This
    lighter memory is model-facing: it keeps short textual feedback and compact
    records that can be serialized into prompts or JSONL logs.
    """

    def __init__(self, top_k: int = 50, feedback_history_size: int = 10):
        self.top_k = int(top_k)
        self.feedback: Deque[LLMFeedbackRecord] = deque(maxlen=int(feedback_history_size))
        self.best_records: List[Dict[str, Any]] = []

    def update(self, evaluation: CandidateEvaluation) -> None:
        """Update recent feedback and, when valid, the top-k memory."""

        self.feedback.append(
            LLMFeedbackRecord(
                reasoning=evaluation.candidate.reasoning,
                valid=evaluation.valid,
                loss=evaluation.loss,
                message=evaluation.message,
            )
        )

        if evaluation.valid and evaluation.loss is not None:
            self.best_records.append(evaluation.to_log_record(include_crn=True))
            self.best_records.sort(key=lambda record: float(record["loss"]))
            if len(self.best_records) > self.top_k:
                self.best_records = self.best_records[: self.top_k]

    def update_many(self, evaluations: Iterable[CandidateEvaluation]) -> None:
        for evaluation in evaluations:
            self.update(evaluation)

    def format_feedback(self, limit: Optional[int] = None) -> str:
        """Return recent feedback as compact prompt text."""

        records = list(self.feedback)
        if limit is not None:
            records = records[-int(limit) :]
        if not records:
            return "No previous LLM feedback is available."
        return "\n".join(record.to_prompt_text() for record in records)

    def format_best(self, limit: int = 5) -> str:
        """Return the best LLM-evaluated candidates as prompt text."""

        records = self.best_records[: int(limit)]
        if not records:
            return "No LLM-generated candidate has been accepted yet."
        lines = []
        for i, record in enumerate(records, start=1):
            candidate = record.get("candidate", {})
            lines.append(
                f"--- LLM best #{i}; loss={float(record['loss']):.6g} ---\n"
                f"reaction_ids={candidate.get('reaction_ids')}\n"
                f"parameter_values={candidate.get('parameter_values')}\n"
                f"CRN:\n{record.get('crn', '')}"
            )
        return "\n".join(lines)
