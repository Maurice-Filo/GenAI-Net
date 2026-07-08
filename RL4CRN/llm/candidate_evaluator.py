"""Evaluate LLM-proposed CRNs through the standard RL4CRN interfaces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from RL4CRN.environments.environment import Environment
from RL4CRN.llm.schemas import CandidateEvaluation, LLMCandidate
from RL4CRN.utils.forbidden_topologies import ForbiddenTopologyArchive


class LLMCandidateEvaluator:
    """Validate and score LLM-generated CRN candidates.

    The evaluator mirrors the RL rollout path: a candidate is converted into raw
    policy-action dictionaries, transformed by the configured actuator, and then
    applied by the configured stepper.  This keeps LLM-assisted search compatible
    with the current RL interfaces instead of constructing CRNs by a separate
    code path.
    """

    def __init__(
        self,
        *,
        crn_template: Any,
        max_added_reactions: int,
        library: Any,
        stepper: Any,
        actuator: Any,
        compute_reward_func: Any,
        is_ordered_policy: bool = False,
        logger: Any = None,
        min_parameter_value: float = 1e-6,
        require_full_budget: bool = True,
        require_unique_reactions: bool = True,
        forbidden_topologies: Optional[ForbiddenTopologyArchive] = None,
        forbidden_loss: float = 1e9,
    ):
        self.crn_template = crn_template
        self.max_added_reactions = int(max_added_reactions)
        self.library = library
        self.stepper = stepper
        self.actuator = actuator
        self.compute_reward_func = compute_reward_func
        self.is_ordered_policy = bool(is_ordered_policy)
        self.logger = logger
        self.min_parameter_value = float(min_parameter_value)
        self.require_full_budget = bool(require_full_budget)
        self.require_unique_reactions = bool(require_unique_reactions)
        self.forbidden_topologies = forbidden_topologies
        self.forbidden_loss = float(forbidden_loss)

    @classmethod
    def from_session(
        cls,
        session: Any,
        *,
        min_parameter_value: float = 1e-6,
        require_full_budget: bool = True,
        require_unique_reactions: bool = True,
    ) -> "LLMCandidateEvaluator":
        """Create an evaluator from a current ``input_interface.Session``."""

        return cls(
            crn_template=session.crn_template,
            max_added_reactions=session.cfg.train.max_added_reactions,
            library=session.library,
            stepper=session.stepper,
            actuator=session.actuator,
            compute_reward_func=session.task.compute_reward,
            is_ordered_policy=bool(getattr(session.cfg.policy, "ordering_enabled", False)),
            logger=session.logger,
            min_parameter_value=min_parameter_value,
            require_full_budget=require_full_budget,
            require_unique_reactions=require_unique_reactions,
            forbidden_topologies=getattr(session, "forbidden_topologies", None),
            forbidden_loss=float(getattr(session.cfg.train, "forbidden_topology_loss", 1e9)),
        )

    def evaluate_many(
        self,
        candidates: Iterable[LLMCandidate],
        *,
        add_to_hall_of_fame: Optional[Any] = None,
        jsonl_path: Optional[str | Path] = None,
    ) -> List[CandidateEvaluation]:
        """Evaluate candidates and optionally store valid ones in a Hall of Fame."""

        evaluations = [self.evaluate(candidate) for candidate in candidates]
        if add_to_hall_of_fame is not None:
            for evaluation in evaluations:
                if (
                    evaluation.valid
                    and evaluation.env is not None
                    and not self.is_forbidden_env(evaluation.env)
                ):
                    add_to_hall_of_fame.add(evaluation.env)
        if jsonl_path is not None:
            self.append_jsonl(evaluations, jsonl_path)
        return evaluations

    def evaluate(self, candidate: LLMCandidate) -> CandidateEvaluation:
        """Validate, materialize, and score one candidate."""

        try:
            prepared = self._prepare_candidate(candidate)
            if isinstance(prepared, CandidateEvaluation):
                return prepared

            reaction_ids, parameter_values = prepared
            env = Environment(
                self.crn_template,
                self.max_added_reactions,
                logger=self.logger,
            )
            env.reset()
            raw_actions = []

            for reaction_id, params in zip(reaction_ids, parameter_values):
                raw_action = {
                    "reaction index": int(reaction_id),
                    "parameters": params,
                    "continuous parameters": params,
                    "discrete parameters": None,
                }
                action = self.actuator.actuate(raw_action)
                if action is None:
                    return CandidateEvaluation(
                        candidate=candidate,
                        valid=False,
                        message=f"Reaction ID {reaction_id} did not resolve to an action.",
                    )
                env.step(action=action, stepper=self.stepper, raw_action=raw_action)
                raw_actions.append(raw_action)

            if self.forbidden_topologies is not None and self.forbidden_topologies.contains_state(env.state):
                return CandidateEvaluation(
                    candidate=candidate,
                    valid=False,
                    loss=self.forbidden_loss,
                    env=env,
                    message="forbidden topology: already archived as evaluated/admissible solution.",
                    raw_actions=raw_actions,
                    task_info={
                        "reward": self.forbidden_loss,
                        "forbidden_topology": True,
                        "source": "LLM",
                    },
                )

            loss, task_info = self._compute_loss(env)
            env.state.last_task_info = dict(getattr(env.state, "last_task_info", {}) or {})
            env.state.last_task_info.update(task_info)
            env.state.last_task_info["reward"] = loss

            return CandidateEvaluation(
                candidate=candidate,
                valid=True,
                loss=loss,
                env=env,
                message="valid",
                raw_actions=raw_actions,
                task_info=env.state.last_task_info,
            )
        except Exception as exc:
            return CandidateEvaluation(
                candidate=candidate,
                valid=False,
                message=f"{type(exc).__name__}: {exc}",
            )

    def _prepare_candidate(
        self, candidate: LLMCandidate
    ) -> tuple[List[int], List[List[float]]] | CandidateEvaluation:
        reaction_ids = list(candidate.reaction_ids)
        parameter_values = [list(params) for params in candidate.parameter_values]

        if len(reaction_ids) != len(parameter_values):
            return CandidateEvaluation(
                candidate=candidate,
                valid=False,
                message="reaction_ids and parameter_values have different lengths.",
            )

        if self.require_full_budget and len(reaction_ids) != self.max_added_reactions:
            return CandidateEvaluation(
                candidate=candidate,
                valid=False,
                message=(
                    f"candidate uses {len(reaction_ids)} reactions; "
                    f"expected {self.max_added_reactions}."
                ),
            )

        if len(reaction_ids) > self.max_added_reactions:
            return CandidateEvaluation(
                candidate=candidate,
                valid=False,
                message=(
                    f"candidate uses {len(reaction_ids)} reactions; "
                    f"budget is {self.max_added_reactions}."
                ),
            )

        if not reaction_ids:
            return CandidateEvaluation(candidate=candidate, valid=False, message="empty candidate.")

        if self.require_unique_reactions and len(set(reaction_ids)) != len(reaction_ids):
            return CandidateEvaluation(
                candidate=candidate,
                valid=False,
                message="candidate contains duplicate reaction IDs.",
            )

        checked_params: List[List[float]] = []
        for reaction_id, params in zip(reaction_ids, parameter_values):
            if reaction_id < 0 or reaction_id >= len(self.library):
                return CandidateEvaluation(
                    candidate=candidate,
                    valid=False,
                    message=f"reaction ID {reaction_id} is outside the library.",
                )

            reaction = self.library.get_reaction(int(reaction_id))
            expected = int(getattr(reaction, "num_parameters", len(params)))
            if len(params) != expected:
                return CandidateEvaluation(
                    candidate=candidate,
                    valid=False,
                    message=(
                        f"reaction ID {reaction_id} expects {expected} parameters; "
                        f"got {len(params)}."
                    ),
                )

            sanitized = []
            for value in params:
                value = float(value)
                if not math.isfinite(value):
                    return CandidateEvaluation(
                        candidate=candidate,
                        valid=False,
                        message="parameters must be finite numbers.",
                    )
                sanitized.append(max(self.min_parameter_value, value))
            checked_params.append(sanitized)

        if self.is_ordered_policy:
            paired = sorted(zip(reaction_ids, checked_params), key=lambda item: item[0])
            reaction_ids = [item[0] for item in paired]
            checked_params = [item[1] for item in paired]

        return [int(idx) for idx in reaction_ids], checked_params

    def _compute_loss(self, env: Environment) -> tuple[float, dict]:
        result = self.compute_reward_func(env.state)
        task_info = {}
        raw_loss = result

        if isinstance(result, (tuple, list)):
            if len(result) == 0:
                raise ValueError("compute_reward_func returned an empty sequence.")
            raw_loss = result[0]
            if len(result) > 1 and isinstance(result[1], dict):
                task_info = dict(result[1])

        # Torch is an optional runtime dependency here from the point of view of
        # this module.  Use duck typing so tests and lightweight installs do not
        # need to import it.
        if hasattr(raw_loss, "detach"):
            raw_loss = raw_loss.detach()
        if hasattr(raw_loss, "cpu"):
            raw_loss = raw_loss.cpu()
        if hasattr(raw_loss, "item"):
            raw_loss = raw_loss.item()
        if isinstance(raw_loss, np.generic):
            raw_loss = raw_loss.item()

        return float(raw_loss), task_info

    def is_forbidden_env(self, env: Any) -> bool:
        """Return True when an evaluated environment matches the forbidden archive."""

        return (
            self.forbidden_topologies is not None
            and env is not None
            and self.forbidden_topologies.contains_state(env.state)
        )

    @staticmethod
    def append_jsonl(evaluations: Sequence[CandidateEvaluation], path: str | Path) -> None:
        """Append evaluation records to a JSONL file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for evaluation in evaluations:
                handle.write(json.dumps(evaluation.to_log_record(), sort_keys=True) + "\n")
