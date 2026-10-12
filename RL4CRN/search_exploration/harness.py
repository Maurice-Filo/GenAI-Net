"""Harness-compatible evaluator that shares the cache and numerical budget."""

from dataclasses import dataclass
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional
from RL4CRN.llm.schemas import CandidateEvaluation
from .evaluation import materialize
from RL4CRN.llm.harness_client import HarnessLLMClient


class ExplorationHarnessClient(HarnessLLMClient):
    """Add a frozen workspace file only when outward communication is permitted."""

    exploration_context = None

    def fork(self):
        client = super().fork()
        client.exploration_context = deepcopy(self.exploration_context)
        return client

    @contextmanager
    def run(self, **kwargs):
        if self.exploration_context is not None:
            kwargs["workspace_files"] = {**dict(kwargs.get("workspace_files") or {}),
                "CONTEXT/EXPLORATION_STATE.json": self.exploration_context}
        with super().run(**kwargs) as workspace:
            yield workspace


@dataclass(frozen=True)
class CachedEvaluation(CandidateEvaluation):
    exploration_record: Optional[dict] = None


class CachedHarnessEvaluator:
    def __init__(self, evaluator, service):
        self.base, self.service = evaluator, service

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evaluate_many(self, candidates, **kwargs):
        return [self.evaluate(c) for c in candidates]

    def evaluate(self, candidate):
        prepared = self.base._prepare_candidate(candidate)
        if isinstance(prepared, CandidateEvaluation):
            return prepared
        ids, parameters = prepared
        values = {"reaction_ids": list(ids), "rates": [float(p[0]) for p in parameters]}
        try:
            record, hit = self.service.evaluate(values, category="llm")
            env = materialize(self.base, values)
            env.state.last_task_info = {**record["task_info"], "reward": record["loss"], "source": "LLM"}
            evaluated = CachedEvaluation(candidate=candidate, valid=True, env=env, loss=record["loss"],
                message="valid; cached" if hit else "valid", task_info=record["task_info"], raw_actions=env.raw_actions_taken,
                exploration_record={**values, **record})
            return evaluated
        except Exception as error:
            return CandidateEvaluation(candidate=candidate, valid=False,
                message=f"{type(error).__name__}: {error}")
