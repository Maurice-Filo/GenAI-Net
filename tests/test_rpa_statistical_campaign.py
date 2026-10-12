import copy
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest
from scipy.stats import wilcoxon

from comparisons.search_exploration.rpa_statistical_campaign import (
    ARMS, BUDGET, SEEDS, configuration, dispatchable, jobs_for,
)
from comparisons.search_exploration.analyze_rpa_statistical import endpoint, holm, paired, summarize


def test_plan_has_matched_seeds_budget_and_local_only_commands(tmp_path):
    jobs = jobs_for(tmp_path, Path("/julia/project"))
    assert len(jobs) == 60
    assert {(j["arm"], j["seed"]) for j in jobs} == {(a, s) for a in ARMS for s in SEEDS}
    for job in jobs:
        command = job["command"]
        assert "--comet-project" not in command
        assert "--post-prune" not in command
        if job["arm"] == "ga":
            assert "--no-comet" in command
            assert command[command.index("--seed")+1] == str(job["seed"])
            assert int(command[command.index("--budget")+1]) == BUDGET
        else:
            assert "--julia-project" in command if job["arm"] == "julia" else "--julia-project" not in command


def test_dispatch_never_exceeds_worker_or_lane_limit(tmp_path):
    jobs = jobs_for(tmp_path, Path("/julia"))
    first = dispatchable(jobs, [])
    assert len(first) == 2 and sum(j["workers"] for j in first) == 64
    assert {j["lane"] for j in first} == {"hybrid", "ga"}
    for job in first:
        job["status"] = "running"
    assert dispatchable(jobs, first) == []
    assert [j["lane"] for j in dispatchable(jobs, [first[1]])] == ["hybrid"]


def test_configuration_changes_are_scoped_and_reference_is_untouched():
    ref = {"schema": "search-streams-v2", "task": "rpa", "policy": {"deep_layer_size": 1024},
        "config": {"seed": 0, "epochs": 100, "batch_size": 1023, "fit_evaluations": 16,
            "evaluation_budget": 1636800, "rate_min": .001, "imitation_weight": .1}}
    original = copy.deepcopy(ref)
    a, b = configuration(ref, 1, "julia"), configuration(ref, 1, "evolution")
    assert ref == original
    assert a["evaluation_budget"] == b["evaluation_budget"] == BUDGET
    assert BUDGET - a["epochs"] * a["batch_size"] == 112493
    assert {k for k in a if a[k] != b[k]} == {"fit_evaluations"}
    with pytest.raises(ValueError):
        configuration(ref, 0, "julia")


def test_statistics_match_exact_wilcoxon_and_handle_all_ties():
    a, b = np.array([11, 12, 13, 14, 15, 16.]), np.full(6, 10.)
    result = paired(a, b, smaller=False, resamples=100)
    assert result["p"] == wilcoxon(a-b, method="exact").pvalue
    assert result["wins"] == 6 and result["ties"] == 0
    tied = paired([3, 3, 3], [3, 3, 3], smaller=False, resamples=100)
    assert tied["p"] == 1 and tied["ties"] == 3
    assert tied["mean_difference_ci95"] == [0., 0.]
    zero = paired([0, 1], [0, 1], smaller=True, resamples=100)
    assert zero["median_paired_ratio"] is None


def test_holm_and_pairwise_direction_and_no_interim_inference():
    assert holm([.03, .01, .04, .2]) == pytest.approx([.09, .04, .09, .2])
    result = paired([1, 2, 3], [2, 2, 1], smaller=True, inference=False)
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
    assert "p" not in result and "mean_difference_ci95" not in result


def test_inference_requires_every_planned_arm_and_seed():
    plan = {"arms": list(ARMS), "seeds": [1, 2], "contrasts": [["julia", "evolution"], ["julia", "ga"]],
        "bootstrap": {"resamples": 100, "seed": 7}}
    rows = [{"arm": a, "seed": s, "status": "complete", "loss": float(i+s), "qualified_discovered": i+s}
        for i, a in enumerate(ARMS) for s in [1, 2]]
    complete = summarize(rows, plan)
    assert complete["inference_ready"]
    assert len(complete["contrasts"]) == 4
    assert all("p_holm" in c for c in complete["contrasts"])
    rows[-1]["status"] = "failed"
    partial = summarize(rows, plan)
    assert not partial["inference_ready"]
    assert all("p" not in c for c in partial["contrasts"])


def test_endpoint_counts_historical_discoveries_not_archive_capacity(tmp_path):
    folder = tmp_path / "runs/ga-seed01"
    folder.mkdir(parents=True)
    manifest = {"schema": "standalone-ga-bounded-v1", "task": "rpa", "harness": False,
        "config": {"seed": 1, "evaluation_budget": BUDGET, "quality_threshold": .02,
            "rate_min": .001, "rate_max": 100., "batch_size": 1023}}
    completed = {"status": "complete", "epochs": 215, "pending_optimization_jobs": 0,
        "elapsed_seconds": 60, "counters": {"expensive_evaluations": BUDGET},
        "metrics": {"Archive/best_loss": .001, "Archive/qualified_topologies": 1}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "completed.json").write_text(json.dumps(completed))
    with sqlite3.connect(folder / "exploration.sqlite") as db:
        db.execute("CREATE TABLE designs(payload TEXT)")
        db.executemany("INSERT INTO designs VALUES (?)", [(json.dumps({"qualified": True, "topology_hash": x}),)
            for x in ("A", "B", "A", "C")])
    plan = {"attempt_ceiling": BUDGET, "threshold": .02}
    job = {"name": folder.name, "arm": "ga", "seed": 1}
    row = endpoint(tmp_path, job, plan, {"status": "complete"})
    assert row["status"] == "complete"
    assert row["qualified_discovered"] == 3 and row["qualified_retained"] == 1
    assert row["canonical_evaluations"] == BUDGET and row["joint_evaluations"] == 0
    manifest["config"]["seed"] = 0
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert endpoint(tmp_path, job, plan, {"status": "complete"})["status"] == "invalid"


def test_ga_local_cli_does_not_import_comet(monkeypatch):
    from comparisons.search_exploration import run_standalone_ga
    from comparisons.search_exploration import freeze_legacy
    monkeypatch.setitem(sys.modules, "comet_ml", None)
    monkeypatch.setattr(sys, "argv", ["run", "--reference", "/missing", "--output", "/missing",
        "--seed", "1", "--budget", "50", "--no-comet"])
    def stop_before_evaluation():
        raise RuntimeError("local configuration reached")
    monkeypatch.setattr(freeze_legacy, "verify", stop_before_evaluation)
    with pytest.raises(RuntimeError, match="local configuration reached"):
        run_standalone_ga.main()


def test_controller_holds_without_dispatch_on_frozen_source_change(tmp_path, monkeypatch):
    from comparisons.search_exploration import rpa_statistical_campaign as controller
    from comparisons.search_exploration import analyze_rpa_statistical as analysis
    jobs = jobs_for(tmp_path, Path("/julia"))
    def changed(*args):
        raise ValueError("Frozen input changed")
    monkeypatch.setattr(controller, "check_frozen", changed)
    monkeypatch.setattr(analysis, "analyze", lambda root: None)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not dispatch"))
    controller.execute(tmp_path, {"jobs": jobs})
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["phase"] == "held"
    assert progress["counts"] == {"queued": 60, "running": 0, "complete": 0, "failed": 0}


def test_controller_preserves_failure_and_holds_remaining_queue(tmp_path, monkeypatch):
    from comparisons.search_exploration import rpa_statistical_campaign as controller
    from comparisons.search_exploration import analyze_rpa_statistical as analysis
    (tmp_path / "logs").mkdir()
    jobs = jobs_for(tmp_path, Path("/julia"))
    class FailedProcess:
        returncode = 1
        pid = 12345
        def poll(self):
            return self.returncode
    monkeypatch.setattr(controller, "check_frozen", lambda *a: None)
    monkeypatch.setattr(analysis, "analyze", lambda root: None)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **k: FailedProcess())
    controller.execute(tmp_path, {"jobs": jobs})
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["phase"] == "held"
    assert progress["counts"] == {"queued": 58, "running": 0, "complete": 0, "failed": 2}
