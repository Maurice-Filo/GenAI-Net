"""Runtime checks for the bidirectional LLM/Hall-of-Fame/SIL bridge."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch

from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.prompts import build_candidate_generation_prompt
from RL4CRN.llm.schemas import LLMCandidate
from RL4CRN.environments.serial_environments import SerialEnvironments
from RL4CRN.utils.input_interface import make_session_and_trainer


def audit_sil_bridge(
    *,
    task_name: str,
    config: Mapping[str, Any],
    build_components: Callable[[dict[str, Any]], tuple[Any, Any, Any, Any]],
) -> dict[str, Any]:
    """Exercise LLM -> shared HoF -> SIL and shared HoF -> LLM communication."""

    mutable_config = _deep_plain_copy(config)
    crn, library_components, task, cfg = build_components(mutable_config)
    search = mutable_config["search"]
    cfg.train.seed = 0
    cfg.train.max_added_reactions = int(search["max_added_reactions"])
    cfg.train.batch_size = 2
    cfg.train.n_cpus = 1
    cfg.train.hall_of_fame_size = cfg.train.max_added_reactions
    trainer = make_session_and_trainer(cfg, task, device="cpu", logger=None)
    session = trainer.s

    library = library_components[0]
    rate_min, rate_max = search["rate_constant_range"]
    null_reaction_id = library.find_zero_reaction()
    evaluator = LLMCandidateEvaluator.from_session(
        session,
        min_parameter_value=float(rate_min),
        max_parameter_value=float(rate_max),
        enforce_parameter_bounds=True,
        forbidden_reaction_ids=[null_reaction_id],
    )
    session.mult_env.reset()
    initial_observation = session.mult_env.observe(session.observer, session.tensorizer)[0]
    initial_reaction_mask = initial_observation[: session.M].bool().detach().cpu().tolist()
    candidate = _make_audit_candidate(
        evaluator,
        initially_masked=initial_reaction_mask,
    )
    hall_of_fame = session.mult_env.hall_of_fame
    evaluation = evaluator.evaluate_many(
        [candidate],
        add_to_hall_of_fame=hall_of_fame,
    )[0]
    if not evaluation.valid or evaluation.loss is None:
        raise RuntimeError(f"{task_name} SIL audit candidate failed: {evaluation.message}")
    if len(hall_of_fame) != 1:
        raise RuntimeError(f"{task_name} LLM candidate did not enter the shared Hall of Fame.")

    stored = hall_of_fame[0]
    raw_action_count = len(stored.raw_actions_taken)
    if raw_action_count != cfg.train.max_added_reactions:
        raise RuntimeError(
            f"{task_name} Hall-of-Fame entry retained {raw_action_count} raw actions; "
            f"expected {cfg.train.max_added_reactions}."
        )

    replay_envs = SerialEnvironments([stored.clone()], hall_of_fame_size=0, logger=None)
    replay_envs.reset()
    replay_log_probabilities = []
    for action_index in range(cfg.train.max_added_reactions):
        observations = replay_envs.observe(session.observer, session.tensorizer)
        raw_actions = [stored.get_raw_action(action_index)]
        log_probabilities = session.policy(observations, mode="full", action=raw_actions)
        replay_log_probabilities.append(float(log_probabilities[0].detach().cpu()))
        replay_envs.step([stored.get_action(action_index)], session.stepper)
    if not all(math.isfinite(value) for value in replay_log_probabilities):
        raise RuntimeError(
            f"{task_name} Hall-of-Fame trajectory is masked during RL SIL replay: "
            f"{replay_log_probabilities}"
        )

    current_batch_loss = torch.tensor(
        [float(evaluation.loss) + 1.0],
        device=session.agent.device,
        dtype=torch.float32,
    )
    sil_loss = session.agent.self_imitation_learingin_loss(
        hall_of_fame,
        current_batch_loss,
        torch.tensor([0], device=session.agent.device),
        observer=session.observer,
        tensorizer=session.tensorizer,
        stepper=session.stepper,
        sil_batch_size=1,
    )
    sil_loss_value = float(sil_loss.detach().cpu()) if hasattr(sil_loss, "detach") else float(sil_loss)
    if not math.isfinite(sil_loss_value):
        raise RuntimeError(f"{task_name} RL SIL replay produced a non-finite loss.")

    prompt = build_candidate_generation_prompt(
        task_description=f"Audit {task_name} communication.",
        reaction_library=session.library,
        max_added_reactions=cfg.train.max_added_reactions,
        num_candidates=1,
        hall_of_fame_iter=hall_of_fame,
    )
    hof_visible_to_llm = "Hall-of-Fame #1" in prompt and str(stored.state) in prompt
    if not hof_visible_to_llm:
        raise RuntimeError(f"{task_name} shared Hall of Fame was not rendered into the LLM prompt.")

    return {
        "task": task_name,
        "solver": str(getattr(crn, "solver", "")),
        "shared_hof_identity": evaluator.crn_template is session.crn_template,
        "llm_to_hof": True,
        "hof_to_llm": hof_visible_to_llm,
        "hof_to_rl_sil": True,
        "raw_action_count": raw_action_count,
        "replay_log_probabilities": replay_log_probabilities,
        "audit_candidate_loss": float(evaluation.loss),
        "sil_loss": sil_loss_value,
    }


def _make_audit_candidate(
    evaluator: LLMCandidateEvaluator,
    *,
    initially_masked: list[bool] | None = None,
) -> LLMCandidate:
    masked = initially_masked or [False] * len(evaluator.library)
    if len(masked) != len(evaluator.library):
        raise ValueError("initially_masked must match the reaction-library size.")
    reaction_ids = [
        reaction_id
        for reaction_id in range(len(evaluator.library))
        if reaction_id not in evaluator.forbidden_reaction_ids and not masked[reaction_id]
    ][: evaluator.max_added_reactions]
    if len(reaction_ids) != evaluator.max_added_reactions:
        raise RuntimeError("Reaction library is too small for the SIL audit candidate.")
    parameter_values = [
        [1.0] * int(evaluator.library.get_reaction(reaction_id).num_parameters)
        for reaction_id in reaction_ids
    ]
    return LLMCandidate(reaction_ids=reaction_ids, parameter_values=parameter_values)


def _deep_plain_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_plain_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_plain_copy(item) for item in value]
    return value
