from dataclasses import replace

from comparisons.llm_seeded_evolution.search import Candidate, Evolution, ScoredCandidate
from comparisons.search_exploration.bounded_ga import BoundedEvolution
from comparisons.search_exploration.analyze_streams import method_label


def test_bounded_wrapper_matches_original_when_no_rate_crosses_bounds():
    kwargs = dict(seed=0, allowed_ids=range(12), n_reactions=5, archive_size=30)
    original = Evolution(**kwargs)
    bounded = BoundedEvolution(bounds=(1e-30, 1e30), **kwargs)
    for i in range(100):
        a, b = original.propose(), bounded.propose()
        assert a == b
        score = ScoredCandidate(a, sum(a.rates), str(sorted(a.reaction_ids)))
        original.archive.add(score)
        bounded.archive.add(replace(score, candidate=b))
    assert bounded.clipped_rates == 0
    assert bounded.archive.ranked() == original.archive.ranked()


def test_bounded_wrapper_clips_without_changing_structure_or_parent(monkeypatch):
    proposal = Candidate((1, 2), (1e-6, 200.), "rate_mutation", "parent")
    monkeypatch.setattr(Evolution, "propose", lambda self: proposal)
    bounded = BoundedEvolution(bounds=(.001, 100.), seed=0, allowed_ids=range(4), n_reactions=2, archive_size=30)
    assert bounded.propose() == replace(proposal, rates=(.001, 100.))
    assert bounded.clipped_rates == 2


def test_control_labels_are_distinct():
    arms = ["rl-mutation-random", "mutation-random-no-rl", "rl-random-no-mutation", "standalone-ga"]
    labels = [method_label({"proposal_arm": arm, "method": "search-streams-v2"}) for arm in arms]
    assert len(set(labels)) == len(arms)


def test_discovery_curve_counts_evicted_topologies_once(tmp_path):
    import json
    import sqlite3
    from comparisons.search_exploration.analyze_streams import read_run
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "standalone-ga-bounded-v1",
        "proposal_arm": "standalone-ga", "task": "rpa", "config": {"quality_threshold": .02}}))
    (tmp_path / "status.json").write_text(json.dumps({"status": "running"}))
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps({"epoch": i,
        "Archive/qualified_topologies": 1, "Compute/expensive_evaluations": i + 1}) for i in range(3)))
    with sqlite3.connect(tmp_path / "exploration.sqlite") as db:
        db.execute("CREATE TABLE designs(key TEXT PRIMARY KEY,payload TEXT)")
        db.execute("CREATE TABLE events(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT,created REAL)")
        db.execute("CREATE TABLE evaluation_summaries(key TEXT PRIMARY KEY,payload TEXT)")
        for epoch, key, topology in [(0, "a", "A"), (1, "b", "B"), (2, "c", "A")]:
            db.execute("INSERT INTO designs VALUES (?,?)", (key, json.dumps({"topology_hash": topology, "qualified": True})))
            db.execute("INSERT INTO events VALUES (?, 'archive_insertion', ?, 0)", (epoch,
                json.dumps({"epoch": epoch, "candidate_hash": key, "stream": "standalone_ga", "source": "ga", "gain": 0})))
    summary, rows, _ = read_run(tmp_path)
    assert summary["discovered_qualifying_topologies"] == 2
    assert [r["Discovery/qualified_topologies"] for r in rows] == [1, 2, 2]
    assert [r["Archive/qualified_topologies"] for r in rows] == [1, 1, 1]
