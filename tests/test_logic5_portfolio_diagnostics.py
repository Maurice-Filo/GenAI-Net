from functools import lru_cache
import json

import numpy as np
import pytest

from comparisons.search_exploration import run_logic5_portfolio_diagnostics as runner
from comparisons.search_exploration.analyze_logic5_portfolio_diagnostics import pass_fraction, cluster


def test_numerical_job_canonicalizes_order_preserving_aligned_rates():
    assert runner.candidate_key([2, 1], [4, 3]) == runner.candidate_key([1, 2], [3, 4])
    assert runner.candidate_key([2, 1], [4, 3]) != runner.candidate_key([1, 2], [4, 3])
    assert runner.action_key([2, 1], [4, 3]) == ((1, (3.,)), (2, (4.,)))
    assert runner.action_key([], []) == ()
    with pytest.raises(ValueError, match="equal lengths"):
        runner.candidate_key([1, 2], [3])


def test_pruning_is_exhaustive_and_perturbations_are_not_clipped(monkeypatch):
    from reproduce.diagnostics import logic_pruning as pruning
    targets = (np.arange(16) % 2).reshape(16, 1).astype(float)

    @lru_cache(maxsize=256)
    def replay(actions):
        ok = len(actions) >= 4
        outputs = np.repeat(targets[:, :, None], 1000, axis=2) if ok else np.full((16, 1, 1000), 1e4)
        ids = tuple(a[0] for a in actions)
        assert ids == tuple(sorted(ids))
        # A permutation must not detach a rate from its reaction, even for subsets.
        expected = {5: 100., 6: 2., 7: 3., 8: 4., 9: 5.}
        for rid, (rate,) in actions:
            assert rate / expected[rid] == pytest.approx(1., abs=.101)
        return {"status": "ok" if ok else "solver_failure", "loss": .01 if ok else 1e4,
                "outputs": outputs, "targets": targets, "topology_hash": str(ids), "structure_json": json.dumps(ids)}

    plan = {"threshold": .0427, "epsilons": [.001, .01, .05], "replay_tolerance": {"rtol": 1e-7, "atol": 1e-9},
            "portfolio_radius": 1.1, "winner_radii": [1.01, 1.05, 1.1, 1.25, 1.5]}
    monkeypatch.setattr(pruning, "replay", replay)
    monkeypatch.setattr(runner, "PLAN", plan)
    monkeypatch.setattr(runner, "DIRECTIONS", np.concatenate((np.ones((16, 10)), -np.ones((16, 10)))))
    candidate = {"id": "test", "reaction_ids": [9, 6, 8, 5, 7], "rates": [5., 2., 4., 100., 3.],
                 "saved_orders": [[5, 6, 7, 8, 9]],
                 "loss": .01, "topology_hash": str((5, 6, 7, 8, 9)), "winner": False}
    record = runner.evaluate_candidate(candidate)
    assert len(record["subsets"]) == 32
    assert record["cores"]["0.01"]["retained_count"] == 4
    assert len(record["perturbations"]) == 64
    assert pass_fraction(record, "original") == pass_fraction(record, "core") == 1.
    rates = runner.perturb([5], [100.], 1.1, np.ones(10))
    assert rates[0] == pytest.approx(110.)


def test_order_fallback_aligns_rates_and_projects_subsets(monkeypatch):
    from reproduce.diagnostics import logic_pruning as pruning
    calls = []
    def replay(actions):
        calls.append(actions)
        return {"status": "solver_failure" if len(calls) == 1 else "ok", "loss": 10.}
    monkeypatch.setattr(pruning, "replay", replay)
    value, attempts, ok = runner.replay_with_orders(((1, (3.,)), (2, (4.,))), [[2, 9, 1]])
    assert ok and len(attempts) == 2
    assert calls == [((1, (3.,)), (2, (4.,))), ((2, (4.,)), (1, (3.,)))]
    assert attempts[0]["status"] == "solver_failure"


def test_unfavorable_successful_draw_is_not_retried(monkeypatch):
    from reproduce.diagnostics import logic_pruning as pruning
    calls = []
    def replay(actions):
        calls.append(actions)
        return {"status": "ok", "loss": 999.}
    monkeypatch.setattr(pruning, "replay", replay)
    value, attempts, ok = runner.replay_with_orders(((1, (3.,)), (2, (4.,))), [[2, 1]])
    assert ok and value["loss"] == 999. and len(calls) == 1


def test_unresolved_baseline_is_persistable_not_blocking(monkeypatch):
    from reproduce.diagnostics import logic_pruning as pruning
    @lru_cache(maxsize=256)
    def replay(actions):
        return {"status": "solver_failure", "loss": 656.}
    monkeypatch.setattr(pruning, "replay", replay)
    candidate = {"id": "failed", "reaction_ids": [1, 2], "rates": [3., 4.], "loss": .01,
                 "saved_orders": [[2, 1]]}
    record = runner.evaluate_candidate(candidate)
    assert record["status"] == "complete" and record["baseline_unresolved"]
    assert record["baseline"]["all16_correct"] is None
    assert len(record["baseline"]["order_attempts"]) == 2
    assert record["evaluator_calls"] == 2
    assert not record["perturbations"]
    json.dumps(record, allow_nan=False)


def test_baseline_recovery_requires_reference_agreement(monkeypatch):
    from reproduce.diagnostics import logic_pruning as pruning
    def replay(actions):
        return {"status": "ok", "loss": 9. if actions[0][0] == 1 else .01}
    monkeypatch.setattr(pruning, "replay", replay)
    value, attempts, ok = runner.replay_with_orders(((1, (3.,)), (2, (4.,))), [[2, 1]],
                                                   lambda r: r["loss"] == .01)
    assert ok and value["loss"] == .01
    assert not attempts[0]["accepted"] and attempts[1]["accepted"]


def test_unresolved_analysis_reports_unknowns_without_tests(monkeypatch, tmp_path):
    from comparisons.search_exploration import analyze_logic5_portfolio_diagnostics as analysis
    from comparisons.search_exploration.analyze_logic5_four_arm import ARMS
    plan = {"epsilons": [.001, .01, .05], "winner_radii": [1.1], "replay_audit_note": "test"}
    selection = {"catalog": {}, "candidates": {"unknown": {}}, "runs": [
        {"arm": arm, "seed": seed, "members": [{"candidate": "unknown"}]}
        for arm in ARMS for seed in range(1, 21)]}
    (tmp_path / "records").mkdir()
    (tmp_path / "records/unknown.json").write_text(json.dumps({
        "id": "unknown", "status": "complete", "plan_sha256": "test", "baseline_unresolved": True,
        "baseline": {"status": "unresolved", "order_attempts": []}, "evaluator_calls": 2,
        "subsets": [], "perturbations": []}))
    monkeypatch.setattr(runner, "load_plan", lambda _: (plan, selection))
    monkeypatch.setattr(runner, "sha", lambda _: "test")
    folder = analysis.analyze(tmp_path)
    result = json.loads((folder / "summary.json").read_text())
    assert result["baseline_unresolved"] == 1 and result["tests"] == []
    assert all(r["baseline_unresolved"] == 1 for r in result["rows"])
    assert result["winner_curves"] == []
    assert (folder / "LOGIC5_FUNCTIONAL_FAMILIES.pdf").exists()


def test_failed_draws_remain_in_denominator():
    rows = [{"mode": "original", "radius": 1.1, "draw": i, "status": "ok" if i < 16 else "solver_failure",
             "valid": True} for i in range(32)]
    assert pass_fraction({"perturbations": rows}, "original") == .5
    with pytest.raises(ValueError, match="Missing"):
        pass_fraction({"perturbations": rows[:-1]}, "original")


def test_families_depend_on_complete_structure_not_arbitrary_ids():
    labels, cuts, families = cluster({"a": ("input-modulated", "A"), "b": ("unmodulated", "A"), "c": ("Z",)})
    assert labels["a"] != labels["b"]  # Jaccard distance 2/3 exceeds .60.
    assert len(families) == 3
    assert set(cuts) == {"0.3", "0.45", "0.6", "0.75"}
