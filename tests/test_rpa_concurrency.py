import json
from pathlib import Path

import pytest

from comparisons.search_exploration.rpa_statistical_campaign import dispatchable, jobs_for
from comparisons.search_exploration.scale_rpa_statistical import AdoptedProcess


def test_scaled_dispatch_limits_with_original_pair_already_active(tmp_path):
    jobs = jobs_for(tmp_path, Path("/julia"))
    active = dispatchable(jobs, [])
    for job in active:
        job["status"] = "running"
    added = dispatchable(jobs, active, 128, {"hybrid": 2, "ga": 2})
    assert len(added) == 2
    assert sum(j["workers"] for j in active+added) == 128
    assert [j["lane"] for j in active+added].count("hybrid") == 2
    assert [j["lane"] for j in active+added].count("ga") == 2
    for job in added:
        job["status"] = "running"
    assert dispatchable(jobs, active+added, 128, {"hybrid": 2, "ga": 2}) == []


def test_adoption_uses_original_parent_exit_status_not_only_completed_file(tmp_path):
    progress = tmp_path / "progress.json"
    job = {"name": "julia-seed05", "pid": 123, "status": "running"}
    progress.write_text(json.dumps({"jobs": [job]}))
    adopted = AdoptedProcess(progress, job)
    assert adopted.poll() is None
    job.update(status="failed", returncode=1)
    progress.write_text(json.dumps({"jobs": [job]}))
    assert adopted.poll() == 1
    assert adopted.adopted


def test_adoption_rejects_process_identity_changes(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"jobs": [{"name": "ga", "pid": 456, "status": "running"}]}))
    with pytest.raises(RuntimeError, match="identity"):
        AdoptedProcess(progress, {"name": "ga", "pid": 123}).poll()


def test_adoption_recognizes_clean_completion(tmp_path):
    progress = tmp_path / "progress.json"
    job = {"name": "ga", "pid": 123, "status": "complete", "returncode": 0}
    progress.write_text(json.dumps({"jobs": [job]}))
    assert AdoptedProcess(progress, job).poll() == 0
