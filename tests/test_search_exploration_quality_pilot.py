import json
from pathlib import Path
import subprocess
import sys

from comparisons.search_exploration.nn_size_quality import jobs_for, task_settings
from comparisons.search_exploration.analyze_nn_size_quality import paired_summary
from RL4CRN.search_exploration.config import ExplorationConfig


def test_quality_pilot_is_matched_except_for_size_and_output(tmp_path):
    jobs = jobs_for(tmp_path, list(range(5)), "genai-net-exploration")
    assert len(jobs) == 20
    for seed in range(5):
        for task in ("logic", "rpa"):
            pair = [j for j in jobs if (j["task"], j["seed"]) == (task, seed)]
            assert {j["embedding"] for j in pair} == {1024, 10240}
            paths = []
            for job in pair:
                command = job["command"]
                assert "--harness" not in command
                assert command[command.index("--comet-project")+1] == "genai-net-exploration"
                assert command[command.index("--embedding-size")+1] == str(job["embedding"])
                paths.append(command[command.index("--exploration-config")+1])
            assert paths[0] == paths[1]
            config = ExplorationConfig(**task_settings(task, seed))
            assert (config.epochs, config.batch_size, config.fit_evaluations, config.evaluation_budget) == (20, 32, 16, 10240)
            assert config.workers == 4 and config.archive_capacity == 2000
            assert config.require_all_rows is (task == "logic")


def test_paired_quality_summary_uses_only_complete_pairs_and_reports_missing():
    def row(seed, size, loss, count, status="complete"):
        return {"task": "logic", "seed": seed, "embedding": size, "loss": loss,
                "qualified_topologies": count, "status": status}
    rows = [row(0, 1024, .01, 5), row(0, 10240, .02, 3),
            row(1, 1024, .001, 9), row(1, 10240, .01, 1, "running"),
            row(2, 1024, None, 0), row(2, 10240, .01, 1)]
    summary = paired_summary(rows, "logic")
    assert summary["matched_complete_seeds"] == [0, 2]
    assert summary["unscorable_loss_pairs"] == 1
    assert summary["small_vs_large_loss"] == {"wins": 1, "ties": 0, "losses": 0}
    assert summary["small_vs_large_qualified_topologies"] == {"wins": 1, "ties": 0, "losses": 1}
    assert summary["median_paired_loss_ratio"] == .5


def test_runner_rejects_nonpositive_embedding_before_creating_output(tmp_path):
    output = tmp_path / "invalid"
    result = subprocess.run([sys.executable, "-m", "comparisons.search_exploration.run",
        "--exploration-config", "comparisons/search_exploration/configs/smoke.json",
        "--output", str(output), "--embedding-size", "0"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "must be positive" in result.stderr
    assert not output.exists()
