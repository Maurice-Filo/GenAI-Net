from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from comparisons.llm_seeded_evolution.search import Candidate, EliteArchive, Evolution, ScoredCandidate, run_search
from comparisons.llm_seeded_evolution.replay import export_replay
from comparisons.llm_seeded_evolution.run import validate_replay
from comparisons.rpa_search.scripts.run_genai_julia_20seed_batch import _command, _run_id


def scored(candidate, loss=1.0):
    return ScoredCandidate(candidate, loss, str(sorted(candidate.reaction_ids)))


def optimizer(seed=0, **kwargs):
    return Evolution(seed=seed, allowed_ids=range(5, 15), n_reactions=3, archive_size=3, **kwargs)


def test_archive_keeps_best_parameters_without_counting_permutations():
    archive = EliteArchive(2)
    a = Candidate((5, 6, 7), (1.0, 2.0, 3.0))
    archive.add(scored(a, 3))
    archive.add(scored(Candidate((7, 5, 6), (3.0, 1.0, 2.0)), 2))
    archive.add(scored(replace(a, rates=(0.1, 2.0, 3.0)), 1))
    archive.add(scored(Candidate((8, 9, 10), (1.0,) * 3), 4))
    archive.add(scored(Candidate((11, 12, 13), (1.0,) * 3), 0.5))
    assert [x.loss for x in archive.ranked()] == [0.5, 1]
    assert len(archive.entries) == 2


def test_mutation_preserves_budget_mask_and_parent():
    evo = optimizer(rate_probability=0, topology_probability=1)
    parent = Candidate((5, 6, 7), (1.0, 2.0, 3.0), "llm", proposal_id="p0")
    evo.archive.add(scored(parent))
    for _ in range(100):
        child = evo.propose()
        assert len(set(child.reaction_ids)) == 3
        assert set(child.reaction_ids) <= set(range(5, 15))
        assert len(set(child.reaction_ids) ^ set(parent.reaction_ids)) == 2
        assert child.proposal_id == "p0"
        assert child.parent_hash == scored(parent).topology_hash
    assert parent.rates == (1.0, 2.0, 3.0)


def test_rate_mutation_preserves_topology_and_does_not_clamp_search_rates():
    evo = optimizer(rate_probability=1, topology_probability=0)
    parent = Candidate((5, 6, 7), (1e-5, 1e4, 1e4))
    evo.archive.add(scored(parent))
    child = evo.propose()
    assert child.reaction_ids == parent.reaction_ids
    assert all(x > 0 for x in child.rates)
    assert sum(x != y for x, y in zip(child.rates, parent.rates)) == 1
    assert max(child.rates) > 100


def test_replay_arrives_at_completion_boundary_and_budgets_are_separate():
    events = [{"after_search_evaluations": 5, "launched_epoch": 0, "candidates": [
        {"reaction_ids": [5, 6, 7], "rates": [1.0] * 3, "proposal_id": "first"}]},
        {"after_search_evaluations": 12, "launched_epoch": 20, "candidates": [
            {"reaction_ids": [8, 9, 10], "rates": [1.0] * 3, "proposal_id": "future"}]}]
    def execute():
        batches, records = [], []
        def evaluate_batch(candidates):
            batches.append(len(candidates))
            return [scored(c, 0.1 if c.source == "llm" else 1) for c in candidates]
        counts = run_search(optimizer(), events=events, search_budget=10, batch_size=4,
                            evaluate_batch=evaluate_batch,
                            record=lambda item, n, m: records.append((item, n, m)))
        return counts, batches, records
    counts, batches, records = execute()
    assert counts == (10, 1)
    assert batches == [4, 1, 1, 4, 1]
    assert [n for item, n, _ in records if item.candidate.source == "llm"] == [5]
    assert all(item.candidate.proposal_id != "future" for item, _, _ in records)
    assert execute() == (counts, batches, records)


def test_empty_round_is_not_replaced_and_terminal_event_is_included():
    evo = optimizer()
    events = [{"after_search_evaluations": 0, "launched_epoch": 0, "candidates": []},
              {"after_search_evaluations": 3, "launched_epoch": 1, "candidates": [
                  {"reaction_ids": [5, 6, 7], "rates": [1.0] * 3, "proposal_id": "p"}]}]
    assert run_search(evo, events=events, search_budget=3, batch_size=2,
                      evaluate_batch=lambda cs: [scored(c) for c in cs],
                      record=lambda *args: None) == (3, 1)


def test_incomplete_evaluation_batch_stops_without_fake_progress():
    with pytest.raises(RuntimeError, match="incomplete"):
        run_search(optimizer(), events=[], search_budget=3, batch_size=2,
                   evaluate_batch=lambda _: [], record=lambda *args: None)


def test_replay_validation_detects_loss_or_topology_drift():
    replay = {"events": [{"candidates": [{"reaction_ids": [5, 6, 7], "rates": [1] * 3,
               "proposal_id": "p", "reference_loss": 1, "topology_hash": "[5, 6, 7]"}]}]}
    assert validate_replay(replay, lambda cs: [scored(c) for c in cs])["status"] == "pass"
    with pytest.raises(ValueError, match="no longer reproduces"):
        validate_replay(replay, lambda cs: [scored(c, 2) for c in cs])


def make_source(tmp_path: Path, mode="none", **flags):
    manifest = {"communication_mode": mode, "task": "logic", "seed": 0, "model": "test",
                "proposal_space": {"forbidden_reaction_ids": [0, 1, 2, 3, 4]}, **flags}
    config = {"logic": {"solver": "CVODE"}, "search": {},
              "rl4crn": {"batch_size": 10, "epochs": 3}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(json.dumps(config))
    c = sqlite3.connect(tmp_path / "results.sqlite")
    c.executescript("""
        CREATE TABLE llm_runs (llm_run_id TEXT, launched_epoch INT, completed_epoch INT,
            accepted INT, requested INT, returned INT, rejected INT);
        CREATE TABLE llm_candidates (llm_run_id TEXT, candidate_index INT, valid INT,
            candidate_json TEXT, loss REAL, topology_hash TEXT);
        CREATE TABLE llm_failures (message TEXT);
        CREATE TABLE hof_snapshots (snapshot_id TEXT, boundary TEXT, created_at REAL,
            cumulative_rl_evaluations INT, cumulative_llm_final_evaluations INT,
            cumulative_llm_tool_evaluations INT);
        CREATE TABLE hof_snapshot_entries (snapshot_id TEXT, rank INT, topology_hash TEXT, loss REAL);
        INSERT INTO llm_runs VALUES ('round',0,1,1,10,1,0);
        INSERT INTO hof_snapshots VALUES ('terminal','terminal_boundary',1,30,1,0);
        INSERT INTO hof_snapshot_entries VALUES ('terminal',1,'hash',0.1);
    """)
    c.execute("INSERT INTO llm_candidates VALUES ('round',0,1,?,0.1,'hash')",
              (json.dumps({"reaction_ids": [5, 6, 7], "parameter_values": [[1], [2], [3]]}),))
    c.commit()
    c.close()
    return tmp_path


def test_legacy_none_manifest_exports_actual_completion_not_launch_epoch(tmp_path):
    replay = export_replay(make_source(tmp_path))
    assert replay["communication_mode"] == "none"
    assert replay["expose_state_to_llm"] is False
    assert replay["insert_llm_into_rl"] is False
    assert replay["events"][0]["after_search_evaluations"] == 20
    assert replay["llm_evaluations"] == 1  # no replacement for nine missing members


@pytest.mark.parametrize("mode,flags", [
    ("full", {}), ("llm-to-rl", {}), ("rl-to-llm", {}),
    ("none", {"expose_state_to_llm": True}), ("none", {"insert_llm_into_rl": True}),
])
def test_source_loader_rejects_every_communication_branch(tmp_path, mode, flags):
    with pytest.raises(ValueError, match="No communication"):
        export_replay(make_source(tmp_path, mode, **flags))


def test_explicit_insertion_replay_preserves_candidates_timing_and_source(tmp_path):
    source = make_source(tmp_path, "llm-to-rl", insert_llm_into_rl=True, expose_state_to_llm=False)
    replay = export_replay(source, source_mode="llm-to-rl")
    assert replay["communication_mode"] == "llm-to-rl"
    assert replay["insert_llm_into_rl"] is True
    assert replay["expose_state_to_llm"] is False
    assert replay["events"][0]["after_search_evaluations"] == 20
    assert replay["events"][0]["candidates"][0]["rates"] == [1, 2, 3]


@pytest.mark.parametrize("mode,flags", [
    ("full", {}), ("none", {}), ("llm-to-rl", {"expose_state_to_llm": True}),
    ("llm-to-rl", {"insert_llm_into_rl": False}),
])
def test_insertion_opt_in_still_rejects_wrong_or_outward_sources(tmp_path, mode, flags):
    with pytest.raises(ValueError, match="without outward"):
        export_replay(make_source(tmp_path, mode, **flags), source_mode="llm-to-rl")


def test_nonempty_wal_is_never_ignored(tmp_path):
    source = make_source(tmp_path)
    (source / "results.sqlite-wal").write_bytes(b"uncheckpointed")
    with pytest.raises(ValueError, match="nonempty WAL"):
        export_replay(source)


@pytest.mark.parametrize("accepted", [None, 0])
def test_failed_source_round_is_preserved_without_replacement(tmp_path, accepted):
    source = make_source(tmp_path)
    with sqlite3.connect(source / "results.sqlite") as c:
        c.executescript("""DROP TABLE llm_failures;
            CREATE TABLE llm_failures (launched_epoch INT, completed_epoch INT,
                requested INT, returned INT, accepted INT, rejected INT);
            INSERT INTO llm_failures VALUES (2,2,10,0,0,0);""")
        c.execute("UPDATE llm_failures SET accepted=?", (accepted,))
    replay = export_replay(source)
    assert replay["source_failed_rounds"] == 1
    assert replay["failed_events"][0]["launched_epoch"] == 2
    assert replay["failed_events"][0]["accepted"] == accepted
    assert len(replay["events"]) == 1
    assert replay["llm_evaluations"] == 1


def test_1023_offspring_use_batch_start_archive_and_insert_after_completed_epoch():
    evo = optimizer(rate_probability=1, topology_probability=0)
    batches = []
    event = {"after_search_evaluations": 1023, "launched_epoch": 0,
             "candidates": [{"reaction_ids": [5, 6, 7], "rates": [1.0] * 3, "proposal_id": "p"}]}
    def evaluate_batch(cs):
        batches.append(cs)
        return [scored(c, 0.01 if c.source == "llm" else 1) for c in cs]
    assert run_search(evo, events=[event], search_budget=2046, batch_size=1023,
                      evaluate_batch=evaluate_batch, record=lambda *args: None) == (2046, 1)
    assert [len(b) for b in batches] == [1023, 1, 1023]
    assert all(c.source == "random" and c.proposal_id is None for c in batches[0])
    assert all(c.source == "rate_mutation" for c in batches[2])
    assert any(c.proposal_id == "p" for c in batches[2])


@pytest.mark.parametrize("task", ["logic", "rpa", "classifier"])
def test_baseline_is_selectable_by_existing_comparison_runner(task):
    method = "llm_seeded_evolution"
    run_id = _run_id(task, method, 0)
    assert "none_replay" in run_id
    command = _command(task, method, 0, run_id, 4, "unused")
    assert "comparisons.llm_seeded_evolution.run" in command
    assert command[command.index("--workers") + 1] == "4"
    if task == "classifier":
        assert "flash_none" in run_id


def test_classifier_uses_hosted_none_audits(tmp_path, monkeypatch):
    from comparisons.llm_seeded_evolution import replay
    source = make_source(tmp_path)
    row = {"task": "classifier", "seed": 0, "status": "pass", "run_root": str(source),
           "database_sha256": replay.sha256(source / "results.sqlite")}
    for name, rows in [("contract_v2_none_seed0_audit.json", [row]),
                       ("contract_v2_none_seeds1to19_audit.json", [])]:
        (tmp_path / name).write_text(json.dumps({"runs": rows}))
    monkeypatch.setattr(replay, "GENERATED", tmp_path)
    assert replay.source_from_audits("classifier", 0) == source
    row["database_sha256"] = "changed"
    (tmp_path / "contract_v2_none_seed0_audit.json").write_text(json.dumps({"runs": [row]}))
    with pytest.raises(ValueError, match="no longer matches"):
        replay.source_from_audits("classifier", 0)
