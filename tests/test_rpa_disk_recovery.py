from copy import deepcopy

import pytest

from comparisons.search_exploration.recover_rpa_disk import restart_job


def example():
    return {"name": "ga-seed09", "status": "running", "seed": 9,
        "output": "/old/ga-seed09", "log_file": "old.log", "pid": 123,
        "command": ["python", "--seed", "9", "--budget", "214793", "--output", "/old/ga-seed09"],
        "run_status": {"metrics": {"Compute/expensive_evaluations": 100}},
        "previous_attempts": [{"charged_evaluations": 5}]}


def test_restart_preserves_history_and_scientific_command(tmp_path):
    original = example()
    before = deepcopy(original)
    job = restart_job(original, tmp_path, {"expensive_evaluations": 120}, ["ok"])
    assert original == before
    assert job["command"][:-1] == original["command"][:-1]
    assert job["status"] == "queued"
    assert "pid" not in job and "run_status" not in job
    assert job["previous_attempts"][0] == original["previous_attempts"][0]
    assert job["previous_attempts"][-1]["charged_evaluations"] == 120
    assert job["previous_attempts"][-1]["charged_evaluations_are_lower_bound"]


def test_unreadable_database_uses_persisted_status_lower_bound(tmp_path):
    job = restart_job(example(), tmp_path, {}, ["database malformed"])
    assert job["previous_attempts"][-1]["charged_evaluations"] == 100
    assert job["previous_attempts"][-1]["database_check"] == ["database malformed"]


def test_completed_runs_cannot_be_restarted(tmp_path):
    job = example()
    job["status"] = "complete"
    with pytest.raises(ValueError, match="Only interrupted"):
        restart_job(job, tmp_path, {}, [])


@pytest.mark.parametrize("existing", ["output", "log"])
def test_existing_recovery_never_overwritten(tmp_path, existing):
    if existing == "output":
        (tmp_path / "runs/ga-seed09-disk-recovery").mkdir(parents=True)
    else:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs/ga-seed09-disk-recovery.log").touch()
    with pytest.raises(ValueError, match="already exists"):
        restart_job(example(), tmp_path, {}, [])
