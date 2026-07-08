"""Lightweight multi-role debate for LLM-assisted CRN generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.memory import LLMMemory
from RL4CRN.llm.prompts import build_candidate_generation_prompt
from RL4CRN.llm.schemas import CandidateEvaluation, LLMCandidate, LLMGenerationConfig, parse_candidates_payload


@dataclass(frozen=True)
class DebateTranscript:
    """One debate round and its evaluated candidates."""

    narrator: str
    proposer: str
    critic: str
    player_payload: Any
    candidates: List[LLMCandidate]
    evaluations: List[CandidateEvaluation]


class LLMCRNDebate:
    """Provider-neutral multi-role wrapper around the CRN candidate evaluator.

    The debate is intentionally small: three text-only roles summarize the search
    state, propose motifs, and criticize them; the final player is constrained to
    return the standard candidate JSON contract.  This preserves the old workflow
    while keeping all CRN validation in ``LLMCandidateEvaluator``.
    """

    def __init__(
        self,
        *,
        client: Any,
        evaluator: LLMCandidateEvaluator,
        memory: Optional[LLMMemory] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ):
        self.client = client
        self.evaluator = evaluator
        self.memory = memory or LLMMemory(top_k=50, feedback_history_size=15)
        self.generation_config = generation_config or LLMGenerationConfig()

    @classmethod
    def from_session(
        cls,
        *,
        client: Any,
        session: Any,
        memory: Optional[LLMMemory] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        require_full_budget: bool = True,
        require_unique_reactions: bool = True,
    ) -> "LLMCRNDebate":
        evaluator = LLMCandidateEvaluator.from_session(
            session,
            require_full_budget=require_full_budget,
            require_unique_reactions=require_unique_reactions,
        )
        return cls(
            client=client,
            evaluator=evaluator,
            memory=memory,
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
    ) -> DebateTranscript:
        base_prompt = build_candidate_generation_prompt(
            task_description=task_description,
            reaction_library=self.evaluator.library,
            max_added_reactions=self.evaluator.max_added_reactions,
            num_candidates=num_candidates,
            hall_of_fame_iter=hall_of_fame_iter,
            feedback_text=self.memory.format_feedback(),
            llm_best_text=self.memory.format_best(),
        )

        narrator = self._generate_text(
            "Narrator: summarize the task, the Hall of Fame, and the recent failures. "
            "Do not propose JSON candidates.\n\n" + base_prompt
        )
        proposer = self._generate_text(
            "Proposer: suggest mechanistic CRN motifs that could improve the loss. "
            "Do not output final JSON.\n\n"
            f"Narrator analysis:\n{narrator}\n\n{base_prompt}"
        )
        critic = self._generate_text(
            "Critic: identify likely invalid choices, missing feedback lessons, and "
            "overfit motifs. Do not output final JSON.\n\n"
            f"Narrator:\n{narrator}\n\nProposer:\n{proposer}"
        )

        player_prompt = (
            "Player: using the debate below, output only the final JSON object "
            "matching the candidate contract.\n\n"
            f"Narrator:\n{narrator}\n\nProposer:\n{proposer}\n\nCritic:\n{critic}\n\n{base_prompt}"
        )
        player_payload = self.client.generate_json(
            player_prompt,
            generation_config=self.generation_config,
        )
        candidates = parse_candidates_payload(player_payload)
        evaluations = self.evaluator.evaluate_many(
            candidates,
            add_to_hall_of_fame=add_to_hall_of_fame,
            jsonl_path=jsonl_path,
        )
        self.memory.update_many(evaluations)
        return DebateTranscript(
            narrator=narrator,
            proposer=proposer,
            critic=critic,
            player_payload=player_payload,
            candidates=candidates,
            evaluations=evaluations,
        )

    def _generate_text(self, prompt: str) -> str:
        """Generate text using either a text or JSON client protocol."""

        if hasattr(self.client, "generate_text"):
            return str(self.client.generate_text(prompt, generation_config=self.generation_config))
        payload = self.client.generate_json(prompt, generation_config=self.generation_config)
        if isinstance(payload, dict):
            return str(payload.get("text", payload))
        return str(payload)
