import fcntl
import json
from pathlib import Path

from comparisons.search_exploration import watch_completed_storage as watcher


def closed_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    for name in ('completed', 'status'):
        (run / f'{name}.json').write_text('{"status":"complete"}')
    (run / 'exploration.sqlite').touch()
    return run


def test_requires_controller_completion_closed_lock_and_checkpoint(tmp_path):
    run = closed_run(tmp_path)
    assert watcher.readiness(run, 'running')[0] == 'waiting_for_training'
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watcher.readiness(run, 'complete')[0] == 'waiting_for_close'
    wal = run / 'results.sqlite-wal'
    wal.write_bytes(b'pending writes')
    assert watcher.readiness(run, 'complete')[0] == 'waiting_for_close'
    wal.unlink()
    assert watcher.readiness(run, 'complete')[0] in ('ready', 'waiting_for_space')
    assert not (run / 'diagnostic_downsampling.json').exists()


def test_failed_training_and_incomplete_migration_are_never_rewritten(tmp_path):
    run = closed_run(tmp_path)
    assert watcher.readiness(run, 'failed')[0] == 'skipped_failed_training'
    marker = run / 'diagnostic_downsampling.json'
    marker.write_text('{"status":"converting"}')
    assert watcher.worker(run, 1) == 1
    marker.write_text('{"status":"complete"}')
    assert watcher.worker(run, 1) == 0


def test_worker_rechecks_readiness_before_migrating(tmp_path, monkeypatch):
    run = closed_run(tmp_path)
    calls = []
    monkeypatch.setattr(watcher, 'migrate', lambda *a, **k: calls.append(a))
    monkeypatch.setattr(watcher, 'readiness', lambda *a: ('waiting_for_close', 'locked'))
    assert watcher.worker(run, 2) == 75
    assert not calls
    monkeypatch.setattr(watcher, 'readiness', lambda *a: ('ready', ''))
    assert watcher.worker(run, 2) == 0
    assert calls == [(run,)]


def test_discovers_queue_and_resolves_dangling_storage_links(tmp_path):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    link = campaign / 'run01'
    target = tmp_path / 'external_volume' / 'run01'
    link.symlink_to(target)
    job = dict(name='seed01', status='queued', command=['python', 'run.py', '--output', str(link)])
    (campaign / 'plan.json').write_text(json.dumps(dict(jobs=[job])))
    (campaign / 'progress.json').write_text(json.dumps(dict(jobs=[dict(name='seed01', status='complete')])))
    registry = tmp_path / 'queue.json'
    registry.write_text(json.dumps(dict(tasks=[dict(task='dose', campaign=str(campaign)),
        dict(task='osc', campaign='absent', dispatch_enabled=False)])))
    jobs = watcher.discover(registry)
    assert len(jobs) == 1 and jobs[0]['run'] == str(target)
    assert jobs[0]['training_status'] == 'complete'
    assert not target.exists()
