"""Adopt live matched-Qwen workers and apply an explicit concurrency override.

This replaces only the queue controller. Scientific inputs remain hash-verified;
running seeds are adopted, never restarted. The original plan remains immutable.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import requests


def alive(pid, output):
    """Check process identity as well as liveness; zombies are finished."""
    try:
        process = Path('/proc') / str(pid)
        state = (process / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (process / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    if state == 'Z':
        return False
    if str(output).encode() not in command:
        raise RuntimeError(f'PID {pid} no longer belongs to the expected worker')
    return True


def main(root, check=False):
    spec = importlib.util.spec_from_file_location('frozen_campaign', root / 'controller/local_qwen_matched_campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    read, write = campaign.read, campaign.write
    for name, digest in read(root / 'input_hashes.json').items():
        if campaign.sha(root / name) != digest:
            raise RuntimeError(f'Frozen input changed: {name}')
    for task, audit in read(root / 'reference_audit.json').items():
        for name, digest in audit['source_hashes'].items():
            if campaign.sha(root / task / 'source' / name) != digest:
                raise RuntimeError(f'Frozen source changed: {task}/{name}')
    lock = None
    if not check:
        lock = (root / 'controller.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = read(root / 'progress.json')
    jobs, issues = progress['jobs'], progress.get('issues', [])
    planned = {j['name']: j for j in read(root / 'plan.json')['jobs']}
    if len(jobs) != len(planned) or {j['name'] for j in jobs} != set(planned):
        raise RuntimeError('Progress does not match frozen job list')
    for job in jobs:
        for field in ('command', 'output', 'task', 'seed'):
            if job[field] != planned[job['name']][field]:
                raise RuntimeError(f'Job input changed: {job["name"]}/{field}')
        if job['status'] == 'queued' and Path(job['output']).exists():
            raise RuntimeError(f'Unreconciled output for queued job {job["name"]}')
        if job['status'] == 'running':
            alive(job['pid'], job['output'])
    if check:
        print(json.dumps({'verified': len(jobs), 'adoptable': [j['name'] for j in jobs if j['status']=='running']}))
        return
    settings = read(root / 'execution_settings.json')
    record = dict(pid=os.getpid(), time=time.time(), adopted=[j['name'] for j in jobs if j['status']=='running'],
                  max_parallel=settings['max_parallel'], controller_sha256=campaign.sha(Path(__file__)))
    write(root / f'controller_handoff_{os.getpid()}.json', record)
    (root / 'controller.pid').write_text(str(os.getpid()) + '\n')
    children = {}
    session = requests.Session()
    session.trust_env = False
    while True:
        for job in jobs:
            name, run = job['name'], Path(job['output'])
            if job['status'] == 'running':
                if name in children:
                    process, log = children[name]
                    if process.poll() is None:
                        continue
                    log.close()
                    job['exit_code'] = process.returncode
                    del children[name]
                else:
                    if alive(job['pid'], run):
                        continue
                    job['exit_code'] = None
                    job['exit_status_source'] = 'Adopted process: completion/failure markers and closed-file checks'
                complete = run / 'completed.json'
                successful = (complete.exists() and read(complete).get('status') == 'complete'
                              and not (run / 'failed.json').exists() and job.get('exit_code') in (None, 0))
                job.update(status='complete' if successful else 'failed', ended=time.time())
                if not successful:
                    issues.append(dict(run=name, issue='Training failed; original artifacts retained'))
                    continue
            if job['status'] == 'complete' and job.get('storage_exit_code') in (None, 75):
                with (root / 'logs' / f'{name}-storage.log').open('a') as log:
                    result = subprocess.run([str(campaign.PYTHON), '-c',
                        'import sys; from watch_completed_storage import worker; sys.exit(worker(sys.argv[1], 2))', str(run)],
                        cwd=root / 'controller', stdout=log, stderr=subprocess.STDOUT)
                job['storage_exit_code'] = result.returncode
                if result.returncode not in (0, 75):
                    issues.append(dict(run=name, issue='Compaction failed; see storage log'))
                history = read(run / 'llm_history.json') if (run / 'llm_history.json').exists() else []
                job['llm_history_entries'] = len(history)
                issue = dict(run=name, issue='LLM request errors recorded; inspect history')
                if any(h.get('error') for h in history) and issue not in issues:
                    issues.append(issue)
        limit = int(read(root / 'execution_settings.json')['max_parallel'])
        if not 1 <= limit <= 8:
            raise RuntimeError('Concurrency must be between 1 and the authorized limit of 8')
        active = sum(j['status'] == 'running' for j in jobs)
        compacting = any(j['status'] == 'complete' and j.get('storage_exit_code') == 75 for j in jobs)
        guard = 'Waiting for completed-run files to close for compaction' if compacting else None
        if shutil.disk_usage(root).free < 80 * 2**30:
            guard = 'Waiting for 80 GiB free space'
        if active < limit and any(j['status']=='queued' for j in jobs) and guard is None:
            try:
                response = session.get(campaign.ENDPOINT + '/models', timeout=10)
                response.raise_for_status()
                if campaign.MODEL not in [m['id'] for m in response.json()['data']]:
                    raise RuntimeError('Expected model not served')
            except Exception as exc:
                guard = 'Local endpoint unavailable: ' + str(exc)
            if guard is None:
                for job in jobs:
                    if active >= limit:
                        break
                    if job['status'] != 'queued':
                        continue
                    if Path(job['output']).exists():
                        raise RuntimeError('Refusing duplicate run: ' + job['name'])
                    log = (root / 'logs' / (job['name'] + '.log')).open('a')
                    process = subprocess.Popen(job['command'], cwd=root / job['task'] / 'source',
                        env=campaign.env(root, job['task']), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status='running', pid=process.pid, started=time.time())
                    children[job['name']] = process, log
                    active += 1
                    write(root / 'progress.json', dict(updated=time.time(), controller_pid=os.getpid(),
                        max_parallel=limit, guard=None, jobs=jobs, issues=issues))
        write(root / 'progress.json', dict(updated=time.time(), controller_pid=os.getpid(), max_parallel=limit,
                                         guard=guard, jobs=jobs, issues=issues))
        if not any(j['status'] in ('queued', 'running') for j in jobs) and not compacting:
            write(root / 'finished.json', dict(time=time.time(), issues=issues))
            break
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    main(args.root, args.check)
