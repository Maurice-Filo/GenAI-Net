"""Single-agent LLM candidate generation for RL4CRN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.memory import LLMMemory
from RL4CRN.llm.prompts import DEFAULT_SYSTEM_PROMPT, build_candidate_generation_prompt
from RL4CRN.llm.schemas import (
    CandidateEvaluation,
    LLMCandidate,
    LLMGenerationConfig,
    parse_candidates_payload,
)
from RL4CRN.llm.vertex_client import DEFAULT_GEMINI_MODEL, VertexLLMClient


@dataclass(frozen=True)
class LLMGenerationRound:
    """Prompt, parsed candidates, and evaluations from one LLM round."""

    prompt: str
    candidates: List[LLMCandidate]
    evaluations: List[CandidateEvaluation]
    raw_payload: Any
    tool_evaluations: List[Any] = field(default_factory=list)


class LLMCRNGenerator:
    """Provider-neutral generator for LLM-proposed CRN candidates."""

    def __init__(
        self,
        *,
        client: Any,
        evaluator: LLMCandidateEvaluator,
        memory: Optional[LLMMemory] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        generation_config: Optional[LLMGenerationConfig] = None,
    ):
        self.client = client
        self.evaluator = evaluator
        self.memory = memory or LLMMemory()
        self.system_prompt = system_prompt
        self.generation_config = generation_config or LLMGenerationConfig()

    @classmethod
    def from_session(
        cls,
        *,
        client: Any,
        session: Any,
        memory: Optional[LLMMemory] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        generation_config: Optional[LLMGenerationConfig] = None,
        require_full_budget: bool = True,
        require_unique_reactions: bool = True,
    ) -> "LLMCRNGenerator":
        """Build a generator from a configured RL4CRN session."""

        evaluator = LLMCandidateEvaluator.from_session(
            session,
            require_full_budget=require_full_budget,
            require_unique_reactions=require_unique_reactions,
        )
        return cls(
            client=client,
            evaluator=evaluator,
            memory=memory,
            system_prompt=system_prompt,
            generation_config=generation_config,
        )

    def run_round(
        self,
        *,
        task_description: str,
        num_candidates: int = 10,
        hall_of_fame_iter: Optional[Iterable[Any]] = None,
        add_to_hall_of_fame: Optional[Any] = None,
        jsonl_path: Optional[str | Path] = None,
    ) -> LLMGenerationRound:
        """Ask the LLM for candidates, evaluate them, and update memory."""

        prompt = build_candidate_generation_prompt(
            task_description=task_description,
            reaction_library=self.evaluator.library,
            max_added_reactions=self.evaluator.max_added_reactions,
            num_candidates=num_candidates,
            hall_of_fame_iter=hall_of_fame_iter,
            feedback_text=self.memory.format_feedback(),
            llm_best_text=self.memory.format_best(),
            system_prompt=self.system_prompt,
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
        return LLMGenerationRound(
            prompt=prompt,
            candidates=candidates,
            evaluations=evaluations,
            raw_payload=raw_payload,
        )


class VertexCRNGenerator(LLMCRNGenerator):
    """Convenience single-agent generator backed by VertexAI Gemini."""

    def __init__(
        self,
        *,
        project_id: str,
        evaluator: LLMCandidateEvaluator,
        location: str = "europe-west1",
        model_name: str = DEFAULT_GEMINI_MODEL,
        memory: Optional[LLMMemory] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        generation_config: Optional[LLMGenerationConfig] = None,
    ):
        client = VertexLLMClient(
            project_id=project_id,
            location=location,
            model_name=model_name,
        )
        super().__init__(
            client=client,
            evaluator=evaluator,
            memory=memory,
            system_prompt=system_prompt,
            generation_config=generation_config,
        )

    @classmethod
    def from_session(
        cls,
        *,
        project_id: str,
        session: Any,
        location: str = "europe-west1",
        model_name: str = DEFAULT_GEMINI_MODEL,
        memory: Optional[LLMMemory] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        generation_config: Optional[LLMGenerationConfig] = None,
        require_full_budget: bool = True,
        require_unique_reactions: bool = True,
    ) -> "VertexCRNGenerator":
        evaluator = LLMCandidateEvaluator.from_session(
            session,
            require_full_budget=require_full_budget,
            require_unique_reactions=require_unique_reactions,
        )
        return cls(
            project_id=project_id,
            evaluator=evaluator,
            location=location,
            model_name=model_name,
            memory=memory,
            system_prompt=system_prompt,
            generation_config=generation_config,
        )
