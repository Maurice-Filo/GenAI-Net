import itertools
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from comparisons.llm_seeded_evolution.parity12 import queue, replay, run


CONFIG = Path(__file__).resolve().parents[1] / "comparisons/rpa_search/configs/logic_parity_12reactions_300epoch.json"


@pytest.mark.parametrize("epoch,expected", [(0, 1023), (8, 9207), (299, 306900), (300, 306900), (310, 306900)])
def test_availability(epoch, expected):
    assert replay.availability(epoch, 300, 1023) == expected


@pytest.mark.parametrize("epoch", [None, -1])
def test_missing_availability_rejected(epoch):
    with pytest.raises(ValueError):
        replay.availability(epoch, 300, 1023)


def test_strict_configuration():
    config = json.loads(CONFIG.read_text())
    replay.check_config(config)
    config["search"]["max_added_reactions"] = 8
    with pytest.raises(ValueError):
        replay.check_config(config)


def test_dependency_requires_exact_complete_grids():
    state = {"completed": [{"cohort": c, "seed": s} for c in ("local_logic", "flash_logic") for s in range(20)]}
    assert queue.complete_grid(state, 40)
    state["failed"] = [{"seed": 1}]
    assert not queue.complete_grid(state, 40)
    state = {"completed": [{"task": "logic_parity", "seed": s} for s in range(20)]}
    assert queue.complete_grid(state, 20, "logic_parity")
    state["completed"][-1]["seed"] = 0
    assert not queue.complete_grid(state, 20, "logic_parity")
    assert not queue.complete_grid({}, 40)


def test_queue_command_has_no_priority_adjustment():
    command = queue.command(None, {"workers": 4, "comet_project": "genai-net-ga-fixed-proposals"},
                            {"replay": "source.json", "output": "new-run"})
    assert "nice" not in command
    assert command[2] == "comparisons.llm_seeded_evolution.parity12.run"


def info():
    inputs = np.array(list(itertools.product([0, 1], repeat=4)))
    target = inputs.sum(axis=1) % 2
    return {"inputs": inputs, "setpoint": target[:, None],
            "outputs": [np.full((1, 1000), float(t)) for t in target]}


def test_correctness_counts_all_rows():
    data = info()
    assert run.correctness(data)["correct_rows"] == 16
    data["outputs"][0][0, -1] = .5
    assert run.correctness(data)["correct_rows"] == 15


@pytest.mark.parametrize("bad", [np.full((1, 1000), 1e4), np.zeros((1, 999)), np.full((1, 1000), np.nan)])
def test_failed_simulation_not_correct(bad):
    data = info()
    data["outputs"][0] = bad
    result = run.correctness(data)
    assert not result["simulation_complete"]
    assert result["correct_rows"] is None


def test_wrong_truth_table_rejected():
    data = info()
    data["setpoint"][0] = 1
    with pytest.raises(ValueError):
        run.correctness(data)


def test_canonical_parity_evaluator_equivalence():
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import build_evaluator
    from comparisons.llm_seeded_evolution.search import Candidate
    from RL4CRN.llm.schemas import LLMCandidate
    from RL4CRN.utils.results_database import serialize_crn

    config = json.loads(CONFIG.read_text())
    run.init_worker(config)
    canonical = build_evaluator("logic_parity", solver="CVODE", config_path=CONFIG)
    assert run.EVALUATOR.forbidden_reaction_ids == canonical.forbidden_reaction_ids == frozenset(range(5))
    assert run.EVALUATOR.max_added_reactions == canonical.max_added_reactions == 12
    rng = np.random.default_rng(71)
    ids = tuple(int(i) for i in rng.choice(range(5, len(canonical.library)), 12, replace=False))
    rates = tuple(float(x) for x in np.exp(rng.uniform(np.log(.1), np.log(50), 12)))
    expected = canonical.evaluate(LLMCandidate(list(ids), [[r] for r in rates]))
    actual = run.evaluate(Candidate(ids, rates))
    assert expected.valid
    assert actual.loss == expected.loss
    assert actual.topology_hash == serialize_crn(expected.env.state)["topology_hash"]
    assert actual.correct_rows == run.correctness(expected.task_info)["correct_rows"]


@pytest.fixture
def completed_source(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    write(tmp_path / "config.json", json.loads(CONFIG.read_text()))
    write(tmp_path / "run_manifest.json", {
        "task": "logic_parity", "seed": 0, "model": "deepseek-v4-flash",
        "communication_mode": "llm-to-rl", "expose_state_to_llm": False,
        "insert_llm_into_rl": True, "proposal_space": {"forbidden_reaction_ids": list(range(5))}})
    for epoch in range(0, 300, 20):
        workspace = tmp_path / "harness-workspaces" / str(epoch)
        write(workspace / "CONTEXT/SEARCH_STATE.json", {"rl_epoch_at_snapshot": epoch})
        write(workspace / "CONTEXT/HALL_OF_FAME.json", {"entries": []})
        write(workspace / "run_manifest.json", {"workspace_file_sha256": {
            name: replay.sha256(workspace / name)
            for name in ("CONTEXT/SEARCH_STATE.json", "CONTEXT/HALL_OF_FAME.json")}})
    with sqlite3.connect(tmp_path / "results.sqlite") as conn:
        conn.executescript("""
            CREATE TABLE llm_runs(llm_run_id, launched_epoch, completed_epoch, requested, returned, accepted, rejected, created_at);
            CREATE TABLE llm_candidates(llm_run_id, candidate_index, valid, candidate_json, loss, topology_hash);
            CREATE TABLE llm_failures(launched_epoch, completed_epoch, requested, returned, accepted, rejected);
            CREATE TABLE hof_snapshots(snapshot_id, boundary, created_at, cumulative_rl_evaluations,
                                      cumulative_llm_final_evaluations, cumulative_llm_tool_evaluations);
            CREATE TABLE hof_snapshot_entries(snapshot_id, rank, topology_hash, loss, parameters_json, task_info_json);
        """)
        for epoch in range(0, 300, 20):
            if epoch == 20:
                conn.execute("INSERT INTO llm_failures VALUES(20, 30, 10, NULL, NULL, 0)")
                continue
            completed = 305 if epoch == 280 else epoch + 5
            conn.execute("INSERT INTO llm_runs VALUES(?,?,?,?,?,?,?,?)", (str(epoch), epoch, completed, 10, 1, 1, 0, epoch))
            candidate = {"reaction_ids": list(range(5, 17)), "parameter_values": [[1.]] * 12}
            conn.execute("INSERT INTO llm_candidates VALUES(?,?,?,?,?,?)", (str(epoch), 0, 1, json.dumps(candidate), .1, "proposal"))
        conn.execute("INSERT INTO hof_snapshots VALUES(1, 'terminal_boundary', 1, 306900, 14, 0)")
        conn.executemany("INSERT INTO hof_snapshot_entries VALUES(1,?,?,?,?,?)",
                         [(i + 1, str(i), .1, "[]", "{}") for i in range(30)])
    return tmp_path


def test_export_preserves_failed_round_and_terminal_only_reply(completed_source):
    result = replay.export(completed_source, replay.sha256(completed_source / "results.sqlite"))
    assert result["llm_evaluations"] == 14
    assert len(result["failed_events"]) == 1
    assert result["failed_events"][0]["accepted"] is None
    assert result["events"][-1]["terminal_only"]
    assert result["events"][-1]["after_search_evaluations"] == 306900
    assert result["all_exposed_hof_empty"]


def test_export_rejects_budget_mismatch(completed_source):
    with sqlite3.connect(completed_source / "results.sqlite") as conn:
        conn.execute("UPDATE hof_snapshots SET cumulative_llm_final_evaluations=15")
    with pytest.raises(ValueError, match="budgets"):
        replay.export(completed_source, replay.sha256(completed_source / "results.sqlite"))


def test_export_rejects_changed_context(completed_source):
    path = completed_source / "harness-workspaces/0/CONTEXT/HALL_OF_FAME.json"
    path.write_text('{"entries": [{"loss": 0.1}]}')
    with pytest.raises(ValueError, match="Changed saved context"):
        replay.export(completed_source, replay.sha256(completed_source / "results.sqlite"))
