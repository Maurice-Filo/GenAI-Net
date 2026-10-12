"""Isolated CPU replay/gradient check, not a reconstruction of trained policies."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3

import torch

from RL4CRN.environments.environment import Environment
from RL4CRN.environments.serial_environments import SerialEnvironments
from RL4CRN.utils.hall_of_fame import HallOfFame
from RL4CRN.utils.input_interface import Session
from RL4CRN.utils.results_database import serialize_crn
from comparisons.rpa_search.src.common.logic_task import build_parity_logic_components
from comparisons.llm_seeded_evolution.replay import GENERATED


def materialize(session, row):
    env = Environment(session.crn_template, 12)
    parameters = json.loads(row["parameters_json"])
    fixed = set(session.crn_template.gather_reaction_IDs())
    for r in parameters:
        if r["reaction_id"] in fixed:
            continue
        raw = {"reaction index": r["reaction_id"], "parameters": r["parameters"],
               "continuous parameters": r["parameters"], "discrete parameters": None}
        env.step(session.actuator.actuate(raw), session.stepper, raw_action=raw)
    env.state.last_task_info = {"reward": row["loss"], "source": row["emitter"]}
    if len(env.raw_actions_taken) != 12 or serialize_crn(env.state)["topology_hash"] != row["topology_hash"]:
        raise ValueError("Materialization changed topology or action count")
    actual = json.loads(serialize_crn(env.state)["parameters_json"])
    expected_rates = {p["reaction_id"]: p["parameters"] for p in parameters}
    if {p["reaction_id"]: p["parameters"] for p in actual} != expected_rates:
        raise ValueError("Reaction-rate pairing changed")
    return env


def teacher(session, samples):
    envs = SerialEnvironments([s.clone() for s in samples], hall_of_fame_size=0)
    envs.reset()
    full, continuous = [], []
    handle = session.policy.continuous_parameter_generator.register_forward_hook(lambda _m, _a, result: continuous.append(result[1]))
    try:
        for j in range(12):
            observations = envs.observe(session.observer, session.tensorizer)
            actions = [s.get_raw_action(j) for s in samples]
            if any(observations[i, a["reaction index"]] != 0 for i, a in enumerate(actions)):
                raise ValueError("Teacher action is masked as already present")
            full.append(session.policy(observations, mode="full", action=actions))
            envs.step([s.get_action(j) for s in samples], session.stepper)
    finally:
        handle.remove()
    for original, reconstructed in zip(samples, envs.envs):
        if serialize_crn(original.state)["parameters_json"] != serialize_crn(reconstructed.state)["parameters_json"]:
            raise ValueError("SIL reset/replay changed reaction parameters")
    full = torch.stack(full).sum(0)
    continuous = torch.stack(continuous).sum(0)
    return full, full - continuous, continuous


def norm(grads):
    return float(torch.sqrt(sum((g.square().sum() for g in grads if g is not None), torch.tensor(0.))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    schedule = json.loads((GENERATED / "parity12_communication_300epoch_20260909_schedule.json").read_text())
    campaign = Path(schedule["campaigns"]["full"]["audit_command"][2])
    manifest = next((campaign / "runs").glob("*seed0/*/run_manifest.json"))
    config = json.loads((manifest.parent / "config.json").read_text())
    _, _, task, cfg = build_parity_logic_components(config)
    cfg.train.seed = 0
    cfg.train.batch_size = 1
    cfg.train.n_cpus = 1
    cfg.train.max_added_reactions = 12
    cfg.train.hall_of_fame_size = 30
    cfg.policy.width = config["rl4crn"]["policy_width"]
    cfg.policy.depth = config["rl4crn"]["policy_depth"]
    cfg.policy.deep_layer_size = config["rl4crn"]["deep_layer_size"]
    session = Session.from_config(cfg, task, device="cpu", logger=None)
    with sqlite3.connect((manifest.parent / "results.sqlite").as_uri() + "?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        snap = c.execute("SELECT snapshot_id FROM hof_snapshots WHERE boundary='terminal_boundary'").fetchone()
        rows = c.execute("SELECT * FROM hof_snapshot_entries WHERE snapshot_id=? ORDER BY rank", (snap[0],)).fetchall()
    samples = [materialize(session, row) for row in rows]
    hof = HallOfFame(30)
    hof.add_all(samples)
    stored = list(hof)
    if len(stored) != 30:
        raise ValueError("HOF deduplication lost distinct archived designs")
    batch_best = torch.tensor([.38736358284950256])
    top_indices = torch.tensor([0])
    losses = torch.tensor([s.state.last_task_info["reward"] for s in stored])
    advantage = (batch_best[0] - losses).clamp(min=0).detach()
    full, structure, continuous = teacher(session, stored)
    valid = torch.isfinite(full) & (advantage > 0)
    manual = -(full[valid] * advantage[valid]).mean()
    structure_loss = -(structure[valid] * advantage[valid]).mean()
    continuous_loss = -(continuous[valid] * advantage[valid]).mean()
    params = list(session.policy.parameters())
    structure_norm = norm(torch.autograd.grad(structure_loss, params, retain_graph=True, allow_unused=True))
    continuous_norm = norm(torch.autograd.grad(continuous_loss, params, allow_unused=True))
    def sil():
        return session.agent.self_imitation_learingin_loss(hof, batch_best, top_indices,
            observer=session.observer, tensorizer=session.tensorizer, stepper=session.stepper)
    exact = sil()
    if not torch.isclose(exact, manual, rtol=1e-5):
        raise ValueError("Independent decomposition does not match production SIL")
    history = []
    for step in range(args.updates):
        session.agent.optimizer.zero_grad()
        loss = sil()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise ValueError("Nonfinite probe gradient")
        session.agent.optimizer.step()
        history.append({"step": step + 1, "sil_loss_before_update": float(loss.detach()), "gradient_norm_before_clip": float(grad_norm)})
    with torch.no_grad():
        after, after_structure, after_continuous = teacher(session, stored)
        final_sil = float(sil())
    # Sampling zero remains possible if it is not explicitly masked; report rather than alter it.
    initial = session.mult_env.observe(session.observer, session.tensorizer)
    null_id = session.library.find_zero_reaction()
    initial_logits = session.policy.reaction_structure_head(session.policy.encoder(initial))
    dist = torch.distributions.Categorical(logits=initial_logits.masked_fill(initial[:, :session.M].bool(), float('-inf')))
    policy_config = asdict(cfg.policy)
    if not math.isfinite(policy_config["constraint_strength"]):
        policy_config["constraint_strength"] = "infinity"
    report = {"scope": "Fresh CPU policy, saved full/seed0 terminal HOF, imitation-only updates with cached losses; NOT the trained campaign policy",
        "source_run": str(manifest.parent), "ode_evaluations": 0, "new_model_calls": 0,
        "policy_class": type(session.policy).__name__, "policy_config": policy_config,
        "agent_config": asdict(cfg.agent), "library_size": session.M, "reactions_per_design": 12,
        "replayed_designs": len(samples), "finite_positive_terms": int(valid.sum()),
        "mask_and_reaction_rate_replay_checks": "pass", "sil_formula_check": "pass",
        "initial_sil": float(exact.detach()), "final_sil": final_sil,
        "initial_mean_structure_logp": float(structure.detach().mean()),
        "final_mean_structure_logp": float(after_structure.mean()),
        "initial_mean_rate_log_density": float(continuous.detach().mean()),
        "final_mean_rate_log_density": float(after_continuous.mean()),
        "initial_structure_gradient_norm": structure_norm, "initial_rate_gradient_norm": continuous_norm,
        "null_reaction_id": null_id, "null_initial_masked": bool(initial[0, null_id]),
        "null_probability_after_probe": float(dist.probs[0, null_id].detach()),
        "updates": history}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "updates"}, indent=2))


if __name__ == "__main__":
    main()
