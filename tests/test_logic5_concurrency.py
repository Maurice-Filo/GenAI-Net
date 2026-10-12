import copy
import json

import pytest

from comparisons.search_exploration.scale_logic5_campaign import reconcile
from comparisons.search_exploration import scale_logic5_campaign as scale


def example():
    jobs = [{"name": str(i), "command": ["python", str(i)], "output": f"/runs/{i}",
             "seed": i, "arm": "full", "status": "queued"} for i in range(14)]
    plan = {"jobs": jobs}
    progress = {"status": "held", "failed": 0, "jobs": copy.deepcopy(jobs)}
    for i, job in enumerate(progress["jobs"][:10]):
        job.update(status="running", pid=100+i, run_status={"epoch": 4})
    return plan, progress


def test_reconcile_preserves_active_and_queued_jobs_without_mutating_frozen_plan():
    plan, progress = example()
    jobs = reconcile(plan, progress)
    assert sum(j["status"] == "running" for j in jobs) == 10
    assert [j["name"] for j in jobs if j["status"] == "queued"] == ["10", "11", "12", "13"]
    assert jobs[0]["pid"] == 100
    assert all("run_status" not in j for j in jobs)
    assert all(j["status"] == "queued" for j in plan["jobs"])


@pytest.mark.parametrize("field", ["command", "output", "seed", "arm"])
def test_reconcile_rejects_scientific_identity_changes(field):
    plan, progress = example()
    progress["jobs"][0][field] = "changed"
    with pytest.raises(ValueError, match="scientific identity"):
        reconcile(plan, progress)


@pytest.mark.parametrize("status,failed", [("running", 0), ("held", 1)])
def test_reconcile_requires_acknowledged_hold_without_failures(status, failed):
    plan, progress = example()
    progress.update(status=status, failed=failed)
    with pytest.raises(ValueError, match="acknowledge HOLD"):
        reconcile(plan, progress)


@pytest.mark.parametrize("failed,expected_new", [(False, 2), (True, 0)])
def test_dispatch_counts_adopted_jobs_and_preserves_failure_hold(tmp_path, monkeypatch, failed, expected_new):
    plan, progress = example()
    for job in progress["jobs"]:
        job["output"] = str(tmp_path / "runs" / job["name"])
    jobs = copy.deepcopy(progress["jobs"])
    if failed:
        progress["jobs"][0].update(status="failed", returncode=1)
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    (tmp_path / "logs").mkdir()
    (tmp_path / "amendments/concurrency12").mkdir(parents=True)
    monkeypatch.setattr(scale, "digest", lambda path: "frozen")
    monkeypatch.setattr(scale, "verify_inputs", lambda plan: None)
    monkeypatch.setattr(scale, "balance_status", lambda: {"is_available": True})
    monkeypatch.setattr(scale, "request_audit", lambda path: {"balance_affected": False})
    monkeypatch.setattr(scale, "donor_alive", lambda identity: True)
    monkeypatch.setattr(scale.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 100 * 2**30})())
    launched, logs = [], []

    class Process:
        pid = 999

    def launch(command, **kwargs):
        launched.append(command)
        logs.append(kwargs["stdout"])
        assert kwargs["start_new_session"]
        return Process()

    class EndIteration(Exception):
        pass

    def sleep(seconds):
        raise EndIteration

    monkeypatch.setattr(scale.subprocess, "Popen", launch)
    monkeypatch.setattr(scale.time, "sleep", sleep)
    amendment = {"jobs": jobs, "effective_concurrency": 12, "donor_controller": {}, "plan_sha256": "frozen"}
    try:
        with pytest.raises(EndIteration):
            scale.execute(tmp_path, plan, amendment)
        assert len(launched) == expected_new
        result = json.loads((tmp_path / "progress-concurrent.json").read_text())
        assert result["active"] == (9 if failed else 12)
        assert result["status"] == ("held" if failed else "running")
        assert result["failed"] == int(failed)
        assert [j["pid"] for j in result["jobs"][:10]] == list(range(100, 110))
    finally:
        for log in logs:
            log.close()
