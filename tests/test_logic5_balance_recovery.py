import copy
import json

import pytest

from comparisons.search_exploration.recover_logic5_balance import reconcile
from comparisons.search_exploration import recover_logic5_balance as recovery


def inputs(tmp_path):
    jobs = [{"name": f"run{i}", "output": str(tmp_path / "old" / f"run{i}"),
             "command": ["python", "run.py", "--output", str(tmp_path / "old" / f"run{i}")],
             "seed": i, "arm": "full", "status": "queued"} for i in range(80)]
    plan = {"jobs": jobs}
    runtime = copy.deepcopy(jobs)
    for j in runtime[:36]:
        j.update(status="complete", returncode=0)
    for j in runtime[36:48]:
        j.update(status="failed", returncode=-15, outage_stop_requested=True)
    return plan, {"status": "held", "active": 0, "jobs": runtime}


def test_recovery_retains_36_restarts_12_and_dispatches_32_fresh(tmp_path):
    plan, progress = inputs(tmp_path)
    root, spill = tmp_path / "recovery", tmp_path / "spill"
    jobs = reconcile(plan, progress, root, spill)
    assert sum(j["retained"] for j in jobs) == 36
    assert sum(j["status"] == "queued" for j in jobs) == 44
    for i, job in enumerate(jobs):
        old = plan["jobs"][i]
        assert job["seed"] == old["seed"] and job["arm"] == old["arm"]
        assert job["command"][:-1] == old["command"][:-1]
        assert job["command"][-1] == job["output"]
        assert job["output"].startswith(str(tmp_path / "old" if i < 36 else root if i < 48 else spill))
    assert all(j["status"] == "queued" for j in plan["jobs"])


@pytest.mark.parametrize("field,value", [("outage_stop_requested", False), ("returncode", 1)])
def test_does_not_retry_unrelated_failures(tmp_path, field, value):
    plan, progress = inputs(tmp_path)
    progress["jobs"][36][field] = value
    with pytest.raises(ValueError, match="explicit diagnosis"):
        reconcile(plan, progress, tmp_path / "new", tmp_path / "spill")


def test_no_recovery_while_active(tmp_path):
    plan, progress = inputs(tmp_path)
    progress["active"] = 1
    with pytest.raises(ValueError, match="stopped"):
        reconcile(plan, progress, tmp_path / "new", tmp_path / "spill")


def test_no_scientific_command_changes(tmp_path):
    plan, progress = inputs(tmp_path)
    progress["jobs"][40]["command"].append("--different-budget")
    with pytest.raises(ValueError, match="identity"):
        reconcile(plan, progress, tmp_path / "new", tmp_path / "spill")


def test_controller_skips_retained_and_launches_twelve_replacements(tmp_path, monkeypatch):
    plan, progress = inputs(tmp_path)
    root, spill = tmp_path / "new", tmp_path / "spill"
    jobs = reconcile(plan, progress, root, spill)
    root.mkdir()
    (root / "logs").mkdir()
    (root / "plan.json").write_text(json.dumps({"jobs": jobs, "original_root": str(tmp_path / "old"),
                                               "spill_root": str(spill), "concurrency": 12}))
    (root / "plan.sha256").write_text("frozen\n")
    monkeypatch.setattr(recovery, "digest", lambda p: "frozen")
    monkeypatch.setattr(recovery, "verify_inputs", lambda p: None)
    monkeypatch.setattr(recovery, "assert_stopped", lambda p: None)
    monkeypatch.setattr(recovery, "balance_status", lambda: {"is_available": True})
    monkeypatch.setattr(recovery.shutil, "disk_usage", lambda p: type("Disk", (), {"free": 100 * 2**30})())
    launched, logs = [], []

    class Process:
        pid = 999

    def launch(command, **kwargs):
        launched.append(command)
        logs.append(kwargs["stdout"])
        return Process()

    class EndIteration(Exception):
        pass

    def sleep(seconds):
        raise EndIteration

    monkeypatch.setattr(recovery.subprocess, "Popen", launch)
    monkeypatch.setattr(recovery.time, "sleep", sleep)
    try:
        with pytest.raises(EndIteration):
            recovery.execute(root)
        result = json.loads((root / "progress.json").read_text())
        assert (result["active"], result["completed"], result["retained"], result["queued"]) == (12, 36, 36, 32)
        assert launched == [j["command"] for j in jobs[36:48]]
    finally:
        for log in logs:
            log.close()
