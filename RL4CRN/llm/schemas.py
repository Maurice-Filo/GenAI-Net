"""Typed objects used by optional LLM-assisted CRN generation.

The LLM integration deliberately keeps model output separate from RL4CRN's
environment objects.  A language model proposes plain candidate records; the
candidate evaluator is then responsible for validating and executing them through
the same actuator/stepper interfaces used by the RL policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class LLMCandidate:
    """One CRN design proposed by an LLM.

    Attributes:
        reaction_ids: Reaction-library IDs, one per added reaction.
        parameter_values: Per-reaction parameter vectors.  The outer list must
            have the same length as ``reaction_ids``.
        reasoning: Optional short explanation returned by the model.
        metadata: Optional model/provider-specific fields preserved for logging.
    """

    reaction_ids: List[int]
    parameter_values: List[List[float]]
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LLMCandidate":
        """Create a candidate from a JSON-like mapping.

        The method performs light structural normalization only.  Full semantic
        validation, such as reaction ID bounds and parameter vector length, is
        done by ``LLMCandidateEvaluator`` because it needs access to the current
        reaction library.
        """

        reaction_ids = [int(idx) for idx in raw.get("reaction_ids", [])]
        raw_params = raw.get("parameter_values", [])
        parameter_values: List[List[float]] = []
        for params in raw_params:
            if isinstance(params, (list, tuple)):
                parameter_values.append([float(p) for p in params])
            else:
                parameter_values.append([float(params)])

        metadata = dict(raw.get("metadata", {}) or {})
        for key, value in raw.items():
            if key not in {"reaction_ids", "parameter_values", "reasoning", "metadata"}:
                metadata[key] = value

        return cls(
            reaction_ids=reaction_ids,
            parameter_values=parameter_values,
            reasoning=str(raw.get("reasoning", "")),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Result of validating and evaluating one LLM candidate."""

    candidate: LLMCandidate
    valid: bool
    loss: Optional[float] = None
    env: Optional[Any] = None
    message: str = ""
    raw_actions: List[Dict[str, Any]] = field(default_factory=list)
    task_info: Dict[str, Any] = field(default_factory=dict)

    def to_log_record(self, include_crn: bool = True) -> Dict[str, Any]:
        """Return a compact JSONL-friendly record.

        ``env`` is intentionally not serialized.  When available, the CRN string
        is included because it is useful for reproducibility and later manual
        inspection.
        """

        record = {
            "candidate": self.candidate.to_dict(),
            "valid": self.valid,
            "loss": self.loss,
            "message": self.message,
            "raw_actions": _json_safe(self.raw_actions),
            "task_info": _json_safe(self.task_info),
        }
        if include_crn and self.env is not None:
            record["crn"] = str(self.env.state)
        return record


@dataclass(frozen=True)
class LLMGenerationConfig:
    """Provider-independent generation options."""

    temperature: float = 0.9
    response_mime_type: str = "application/json"
    max_output_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return only non-``None`` values for provider adapters."""

        return {k: v for k, v in asdict(self).items() if v is not None}


def parse_candidates_payload(payload: Any) -> List[LLMCandidate]:
    """Parse a model response into ``LLMCandidate`` objects.

    Accepted payloads are either ``{"candidates": [...]}`` or a bare list of
    candidate mappings.  A ``ValueError`` is raised for incompatible shapes; this
    keeps provider errors visible instead of silently returning an empty set.
    """

    if isinstance(payload, Mapping):
        candidates_raw = payload.get("candidates", [])
    elif isinstance(payload, list):
        candidates_raw = payload
    else:
        raise ValueError("LLM response must be a mapping or a list of candidates.")

    if not isinstance(candidates_raw, list):
        raise ValueError("The 'candidates' field must be a list.")

    return [LLMCandidate.from_mapping(item) for item in candidates_raw]


def candidates_to_payload(candidates: Iterable[LLMCandidate]) -> Dict[str, Any]:
    """Serialize candidates using the prompt contract."""

    return {"candidates": [candidate.to_dict() for candidate in candidates]}


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python objects to JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
