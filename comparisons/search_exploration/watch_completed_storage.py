"""Apply the diagnostic storage policy after registered campaign runs close.

This independent watcher works with already-running, frozen campaign controllers.
It remains alive after the current queue finishes and discovers newly registered
campaigns. Failed training runs are retained untouched for diagnosis.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:  # Also runnable as a frozen standalone script beside the migration module.
    from .downsample_closed_diagnostics import atomic_json, assert_no_open_database, migrate
except ImportError:
    from downsample_closed_diagnostics import atomic_json, assert_no_open_database, migrate


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def discover(registry):
    jobs = []
    for task in read(registry)['tasks']:
        if not task.get('campaign') or task.get('dispatch_enabled') is False:
            continue
        campaign = Path(task['campaign'])
        plan = read(campaign / 'plan.json')
        progress = {j['name']: j for j in read(campaign / 'progress.json').get('jobs', [])}
        for job in plan['jobs']:
            command = job['command']
            run = Path(command[command.index('--output') + 1]).resolve()
            jobs.append(dict(run=str(run), task=task['task'], name=job['name'],
                             training_status=progress.get(job['name'], job)['status']))
    if len({j['run'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate run paths in registry')
    return jobs


def readiness(run, training_status):
    """Readiness is checked again in the worker; the migration owns run.lock."""
    run = Path(run)
    marker = read(run / 'diagnostic_downsampling.json')
    if marker:
        if marker.get('status') == 'complete':
            return 'complete', ''
        return 'needs_attention', 'Existing migration record requires reconciliation'
    if training_status == 'failed' or (run / 'failed.json').exists():
        return 'skipped_failed_training', 'Original diagnostics retained'
    if training_status != 'complete':
        return 'waiting_for_training', ''
    if any(read(run / name).get('status') != 'complete' for name in ('completed.json', 'status.json')):
        return 'waiting_for_close', 'Completion markers not ready'
    if not (run / 'exploration.sqlite').exists():
        return 'needs_attention', 'Completed run has no exploration database'
    if (run / 'exploration.downsample-building.sqlite').exists():
        return 'needs_attention', 'Unrecorded temporary database requires reconciliation'
    with (run / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'waiting_for_close', 'Run still holds its lock'
        for name in ('exploration.sqlite', 'results.sqlite'):
            db = run / name
            wal = Path(str(db) + '-wal')
            if wal.exists() and wal.stat().st_size:
                return 'waiting_for_close', f'{name} has uncheckpointed writes'
            try:
                assert_no_open_database(db)
            except RuntimeError:
                return 'waiting_for_close', f'{name} is still open'
    if shutil.disk_usage(run).free < 15 * 2**30:
        return 'waiting_for_space', 'Less than 15 GiB temporary headroom'
    return 'ready', ''


def worker(run, workers):
    status, reason = readiness(run, 'complete')
    if status == 'complete':
        return 0
    if status != 'ready':
        print(json.dumps(dict(status=status, reason=reason)), flush=True)
        return 75 if status.startswith('waiting_') else 1
    try:
        migrate(run, workers=workers)
    except (BlockingIOError, RuntimeError):
        # A reader/writer can open between precheck and migration lock acquisition.
        if not (Path(run) / 'diagnostic_downsampling.json').exists():
            status, _ = readiness(run, 'complete')
            if status.startswith('waiting_'):
                return 75
        raise
    return 0


def watch(registry, state_dir, workers=2, poll_seconds=30):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    issues = read(state_dir / 'issues.json')
    active = None
    while True:
        if active and active['process'].poll() is not None:
            code = active['process'].returncode
            active['log'].close()
            marker = read(Path(active['run']) / 'diagnostic_downsampling.json')
            if code != 75 and (code != 0 or marker.get('status') != 'complete'):
                issues[active['run']] = dict(exit_code=code, recorded=time.time(),
                    error=marker.get('error', 'See migration log; original retained unless marker states otherwise'),
                    log=active['log_path'])
                atomic_json(state_dir / 'issues.json', issues)
            active = None
        jobs = discover(registry)
        for job in jobs:
            run = job['run']
            if active and active['run'] == run:
                job['status'] = 'compacting'
                marker = read(Path(run) / 'diagnostic_downsampling.json')
                job['migration'] = {k: marker[k] for k in ('status', 'processed', 'diagnostic_total') if k in marker}
                continue
            status, reason = readiness(run, job['training_status'])
            if run in issues and status != 'complete':
                status, reason = 'needs_attention', issues[run]['error']
            job.update(status=status, reason=reason)
            if status == 'complete':
                job['reclaimed_bytes'] = read(Path(run) / 'diagnostic_downsampling.json').get('reclaimed_bytes', 0)
            elif status == 'ready' and active is None:
                log_path = state_dir / (job['name'] + '.log')
                log = log_path.open('a')
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--process-run', run, '--workers', str(workers)], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
                active = dict(run=run, process=process, log=log, log_path=str(log_path))
                job.update(status='compacting', pid=process.pid)
        counts = dict(Counter(j['status'] for j in jobs))
        terminal = {'complete', 'skipped_failed_training', 'needs_attention'}
        progress = dict(status='watching' if all(j['status'] in terminal for j in jobs) else 'processing_or_waiting',
            pid=os.getpid(), updated=time.time(), registry=str(registry), counts=counts,
            reclaimed_bytes=sum(j.get('reclaimed_bytes', 0) for j in jobs), jobs=jobs)
        atomic_json(state_dir / 'progress.json', progress)
        time.sleep(poll_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--process-run', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.workers < 1 or args.poll_seconds < 1:
        parser.error('Workers and polling interval must be positive')
    if args.process_run:
        sys.exit(worker(args.process_run, args.workers))
    if not args.registry or not args.state_dir:
        parser.error('--registry and --state-dir are required for watching')
    watch(args.registry.resolve(), args.state_dir.resolve(), args.workers, args.poll_seconds)
