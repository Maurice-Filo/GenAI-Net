"""Run the frozen Flash prompt ablation, compacting completed seeds before dispatch."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


def account_status():
    # Authentication is used only for the configured provider, never written to output.
    import yaml
    credentials = Path('/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home/.credentials.yaml')
    key = str((yaml.safe_load(credentials.read_text()) or {}).get('DEEPSEEK_API_KEY', '')).strip()
    if not key:
        raise RuntimeError('Missing configured DeepSeek credential')
    request = urllib.request.Request('https://api.deepseek.com/user/balance',
                                    headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=20) as response:
        value = json.load(response)
    dollars = sum(float(b['total_balance']) for b in value.get('balance_infos', []) if b['currency']=='USD')
    return dict(available=value['is_available'], balance_usd=dollars, checked=time.time())


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def environment(root):
    env = os.environ.copy()
    env.pop('COMET_API_KEY', None)
    env.update(COMET_MODE='DISABLED', PYTHONPATH=str(root / 'source'), CUDA_VISIBLE_DEVICES='',
               TMPDIR=str(root / 'tmp'), MPLCONFIGDIR=str(root / 'mpl'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JULIA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def main(root, check=False):
    for name, digest in read(root / 'input_hashes.json').items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    subprocess.run([sys.executable, '-m', 'comparisons.search_exploration.freeze_legacy', '--verify'],
                   cwd=root / 'source', env=environment(root), check=True)
    plan = read(root / 'plan.json')
    assert plan['root'] == str(root) and plan['max_parallel'] == 8
    if check:
        print(json.dumps(dict(verified=True, jobs=len(plan['jobs']), parallel=8)))
        return
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'started.json').exists():
        raise RuntimeError('Already launched; explicit reconciliation required')
    write(root / 'started.json', dict(pid=os.getpid(), time=time.time()))
    jobs, issues, active = plan['jobs'], [], {}
    balance, last_balance_check = {}, 0
    while True:
        for name, (proc, log, job) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close()
            run = Path(job['output'])
            done = read(run / 'completed.json') if (run / 'completed.json').exists() else {}
            good = proc.returncode == 0 and done.get('status') == 'complete' and not (run / 'failed.json').exists()
            job.update(status='complete' if good else 'failed', exit_code=proc.returncode, ended=time.time())
            del active[name]
            if not good:
                issues.append(dict(run=name, issue='Run failed; original artifacts retained'))
        for job in jobs:
            if job['status'] != 'complete' or job.get('storage_exit_code') not in (None, 75):
                continue
            with (root / 'logs' / (job['name'] + '-storage.log')).open('a') as log:
                result = subprocess.run([sys.executable, str(root / 'controller/watch_completed_storage.py'),
                    '--process-run', job['output'], '--workers', '2'], stdout=log, stderr=subprocess.STDOUT,
                    env=environment(root))
            job['storage_exit_code'] = result.returncode
            if result.returncode not in (0, 75):
                issues.append(dict(run=job['name'], issue='Compaction failed; see storage log'))
            history_path = Path(job['output']) / 'llm_history.json'
            if history_path.exists():
                history = read(history_path)
                job['llm_requests'] = len(history)
                job['llm_errors'] = sum(bool(h.get('error')) for h in history)
        guard = None
        running_histories = []
        for job in jobs:
            if job['status'] != 'running':
                continue
            history_path = Path(job['output']) / 'llm_history.json'
            if history_path.exists():
                running_histories.extend(read(history_path))
        if (sum(bool(h.get('error')) for h in running_histories) >= 2
                and not any(h.get('valid', 0) for h in running_histories)):
            (root / 'HOLD').write_text('Automatic hold: repeated LLM failures before any valid proposal; inspect request logs.\n')
        if shutil.disk_usage(root).free < 100 * 2**30:
            guard = 'Waiting for 100 GiB free on local1'
        if any(j['status']=='complete' and j.get('storage_exit_code') != 0 for j in jobs):
            guard = 'Waiting for completed-run compaction'
        if (root / 'HOLD').exists():
            guard = 'HOLD file present'
        if any(j['status']=='queued' for j in jobs):
            if time.time() - last_balance_check >= 60:
                try:
                    balance = account_status()
                except Exception as exc:
                    balance = dict(available=False, error=type(exc).__name__, checked=time.time())
                last_balance_check = time.time()
            # Dispatch reserve, not an estimate or a guarantee of total campaign cost.
            if not balance.get('available') or balance.get('balance_usd', 0) < 5:
                guard = (guard + '; ' if guard else '') + 'Waiting for DeepSeek top-up (US$5 dispatch reserve)'
        if not guard:
            for job in jobs:
                if len(active) >= 8:
                    break
                if job['status'] != 'queued':
                    continue
                if Path(job['output']).exists():
                    raise RuntimeError('Refusing duplicate output: ' + job['name'])
                log = (root / 'logs' / (job['name'] + '.log')).open('x')
                proc = subprocess.Popen(job['command'], cwd=root / 'source', env=environment(root),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status='running', pid=proc.pid, started=time.time())
                active[job['name']] = proc, log, job
                print('Started ' + job['name'] + ' pid=' + str(proc.pid), flush=True)
                write(root / 'progress.json', dict(status='running', updated=time.time(),
                    controller_pid=os.getpid(), max_parallel=8, guard=None, account=balance,
                    jobs=jobs, issues=issues))
        state = dict(status='waiting' if guard and not active else 'running', updated=time.time(),
                     controller_pid=os.getpid(), max_parallel=8, guard=guard, account=balance, jobs=jobs, issues=issues)
        write(root / 'progress.json', state)
        if not any(j['status'] in ('queued', 'running') for j in jobs) and all(
            j.get('storage_exit_code')==0 for j in jobs if j['status']=='complete'):
            state['status'] = 'complete'
            write(root / 'progress.json', state)
            write(root / 'finished.json', dict(time=time.time(), issues=issues))
            return
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    main(args.root.resolve(), args.check)
