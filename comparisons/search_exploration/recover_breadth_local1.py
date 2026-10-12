"""Restart explicitly authorized disk-interrupted seeds and queued breadth jobs.

Frozen experiment inputs are verified unchanged. Every successful run is
compacted before a replacement job is launched; failures retain their evidence.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from comparisons.search_exploration.downsample_closed_diagnostics import atomic_json
from comparisons.search_exploration.watch_completed_storage import worker as compact

NAMES = ['dose-ultrasensitive-full-20seed-20261001', 'dose-biphasic-full-20seed-20261001',
         'classifier-full-20seed-20261001', 'parity12-breadth-full-20seed-20261001']


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def verify(root):
    plan = read(root/'plan.json')
    assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest() == (root/'plan.sha256').read_text().strip()
    for base, hashes in [(root/'source', plan['source_hashes']), (root, plan['input_hashes'])]:
        for rel, expected in hashes.items():
            assert hashlib.sha256((base/rel).read_bytes()).hexdigest() == expected, rel
    return plan


def prepare_jobs(root, plan):
    """Preserve interrupted attempts, including their logs, before fresh reruns."""
    previous = read(root/'progress.json')
    by_name = {j['name']: j for j in previous.get('jobs', [])}
    jobs = []
    attempts = root/'interrupted_attempts_20261005'
    for original in plan['jobs']:
        job = dict(original)
        run = root/'runs'/job['name']
        if (run/'completed.json').exists():
            assert read(run/'completed.json').get('status') == 'complete'
            assert read(run/'status.json').get('status') == 'complete'
            job.update(by_name.get(job['name'], {}));job['status'] = 'complete'
        else:
            if run.exists():
                attempts.mkdir(exist_ok=True)
                preserved = attempts/job['name']
                assert not preserved.exists() and not preserved.is_symlink()
                run.rename(preserved)  # Preserve symlink targets as well as local directories.
                log = root/'logs'/f"{job['name']}.log"
                if log.exists():log.rename(attempts/f"{job['name']}.log")
                job['previous_attempt'] = str(preserved)
                job['restart_reason'] = 'Disk-full interruption; fresh run with identical seed and frozen inputs'
            job['status'] = 'queued'
        assert str(run.resolve()).startswith('/local1/rossin/'), run
        jobs.append(job)
    return jobs


def execute(root, state):
    lock = (root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    waiting = (root/'waiting.lock').open('a');fcntl.flock(waiting,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = verify(root)
    marker = root/'recovery_started_20261005.json'
    if marker.exists():raise RuntimeError('Recovery already started; reconcile instead of silently rerunning')
    snapshot = root/'pre_recovery_20261005';snapshot.mkdir(exist_ok=False)
    for name in ['progress.json','issues.json','started.json']:
        if (root/name).exists():shutil.copy2(root/name,snapshot/name)
    jobs = prepare_jobs(root,plan)
    atomic_json(marker,dict(pid=os.getpid(),started=time.time(),jobs=jobs,
        authorization='User requested restart of missing runs on local1 and immediate per-run downsampling',
        scientific_inputs='Unchanged frozen source, prompts, task configs, seeds and budgets'))
    issues = read(root/'issues.json') or []
    env = dict(os.environ,PYTHONPATH=str(root/'source'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JULIA_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mpl'),COMET_MODE='DISABLED')
    env.pop('COMET_API_KEY',None)
    temporary=root/'tmp';temporary.mkdir(exist_ok=True);env['TMPDIR']=str(temporary)
    active={}

    def save(status='running'):
        counts=dict(Counter(j['status'] for j in jobs))
        atomic_json(root/'progress.json',dict(status=status,updated=time.time(),counts=counts,
            jobs=[{**j,'run_status':read(root/'runs'/j['name']/'status.json')} for j in jobs],
            recovery='breadth-recovery-20261005',storage='local1; compact each successful run before dispatching a replacement'))
        atomic_json(root/'issues.json',issues)
        atomic_json(state/'progress.json',dict(status=status,campaign=str(root),updated=time.time(),counts=counts,pid=os.getpid()))

    def downsample(job):
        run=root/'runs'/job['name']
        job['storage_status']='compacting';save()
        try:
            code=compact(run,2)
            if code:raise RuntimeError(f'Compactor returned {code}; see run and migration markers')
            job['storage_status']='complete'
        except Exception as error:
            job['storage_status']='needs_attention'
            issues.append(dict(run=job['name'],kind='storage_compaction',error=str(error)))
        save()

    # Reconcile existing completed runs first (normally already compacted).
    for job in jobs:
        if job['status']=='complete':downsample(job)
    save()
    while True:
        for name,(process,log,job) in list(active.items()):
            if process.poll() is None:continue
            log.close();run=root/'runs'/name
            done=read(run/'completed.json')
            job.update(status='complete' if process.returncode==0 and done.get('status')=='complete' else 'failed',
                       exit_code=process.returncode,finished=time.time())
            del active[name]
            if job['status']=='complete':
                downsample(job)
            else:
                issues.append(dict(run=name,kind='failed_run',exit_code=process.returncode,
                    detail='See failed.json and run log; no automatic retry'))
            # Audit request failures without affecting scientific execution.
            try:
                from comparisons.search_exploration.recover_rpa_directional_balance import request_audit
                audit=request_audit(run);job['request_audit']=audit
                if audit['failed_requests'] or not audit['valid_members']:
                    issues.append(dict(run=name,kind='llm_request_issues',detail=audit))
            except Exception as error:
                issues.append(dict(run=name,kind='request_audit_error',error=str(error)))
            counters=done.get('counters',{})
            numerical={k:v for k,v in counters.items() if v and any(s in k for s in ('disagreement','failed','unknown'))}
            if numerical:issues.append(dict(run=name,kind='numerical_diagnostics',detail=numerical))
        free=shutil.disk_usage(root).free
        for job in jobs:
            if len(active)>=plan['concurrency'] or free<80*2**30:break
            if job['status']!='queued':continue
            log=(root/'logs'/f"{job['name']}.log").open('x')
            process=subprocess.Popen(job['command'],cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job.update(status='running',pid=process.pid,started=time.time())
            active[job['name']]=(process,log,job)
            print('Started '+job['name'],flush=True)
        queued=any(j['status']=='queued' for j in jobs)
        save('waiting_for_storage' if queued and free<80*2**30 else 'running')
        if not active and not queued:break
        time.sleep(5)
    save('finished')
    atomic_json(root/'finished.json',dict(finished=time.time(),counts=dict(Counter(j['status'] for j in jobs)),
        issues=len(issues),recovery='20261005',preserved_attempts=str(root/'interrupted_attempts_20261005')))
    lock.close();waiting.close()


def main(reports,state):
    state.mkdir(parents=True,exist_ok=True)
    lock=(state/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in NAMES:
        root=(reports/name).resolve()
        assert str(root).startswith('/local1/rossin/')
        execute(root,state)
    atomic_json(state/'finished.json',dict(finished=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    a=p.parse_args();main(a.reports.resolve(),a.state.resolve())
