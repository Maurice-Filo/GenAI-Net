"""Explicit same-seed recovery after provider balance failure; no automatic retry."""

import argparse
import collections
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

from comparisons.search_exploration.freeze_legacy import ROOT, digest
from comparisons.search_exploration.rpa_directional_stream_campaign import read, external_active, FULL_ROOT

ORIGINAL = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-directional-20seed-20260917')


def balance_error(text):
    text = text.lower()
    return 'insufficient balance' in text or 'insufficient_balance' in text


def balance_status():
    """Only return balance data; never persist or print authentication material."""
    import yaml
    path = Path.home() / 'ai-workspaces/deepseek-test/.dsh-home/.credentials.yaml'
    key = str((yaml.safe_load(path.read_text()) or {}).get('DEEPSEEK_API_KEY', '')).strip()
    if not key:
        raise RuntimeError('Missing protected DeepSeek credential')
    request = urllib.request.Request('https://api.deepseek.com/user/balance',
                                    headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    return {'checked_at': time.time(), 'is_available': data['is_available'],
            'balances': [{k: b.get(k) for k in ('currency', 'total_balance')}
                         for b in data.get('balance_infos', [])]}


def request_audit(folder):
    history = read(folder / 'llm_history.json') or []
    failed, quota = 0, 0
    for row in history:
        error = str(row.get('error', ''))
        if not error:
            continue
        failed += 1
        match = re.search(r'inspect (.+?/stderr.txt)', error)
        stderr = Path(match.group(1)) if match else None
        if balance_error(error) or (stderr is not None and stderr.exists() and balance_error(stderr.read_text())):
            quota += 1
    # An error can precede the host's next collection of its background request.
    stderr_hits = sum(balance_error(p.read_text()) for p in folder.glob('harness-workspaces/*/calls/*/stderr.txt'))
    return {'requests': len(history), 'failed_requests': failed, 'balance_failed_requests': quota,
            'balance_stderr_files': stderr_hits, 'balance_affected': bool(quota or stderr_hits),
            'valid_members': sum(h.get('valid', 0) for h in history)}


def replace_output(command, output):
    new = list(command)
    if new.count('--output') != 1:
        raise ValueError('Expected one output directory')
    new[new.index('--output') + 1] = str(output)
    return new


def verify_inputs(plan):
    for path, expected in {**{str(ROOT / p): h for p, h in plan['source_hashes'].items()},
                           **plan['input_hashes']}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def prepare(original, root, concurrency=10):
    from RL4CRN.search_exploration.engine import atomic_json
    plan = read(original / 'plan.json')
    if read(original / 'progress.json')['status'] != 'complete':
        raise ValueError('Original campaign must be stopped/completed before reconciliation')
    if not 1 <= concurrency <= 10:
        raise ValueError('Concurrency must be between one and ten')
    verify_inputs(plan)
    if shutil.disk_usage(root.parent).free < 100 * 2**30:
        raise OSError('Require at least 100 GiB free')
    jobs = []
    for j in plan['jobs']:
        old = original / 'runs' / j['name']
        c = read(old / 'completed.json')
        if not c or c['status'] != 'complete':
            raise ValueError(f'Original endpoint not complete: {old}')
        audit = request_audit(old)
        affected = audit['balance_affected']
        output = root / 'runs' / (j['name'] + '-balance-recovery') if affected else old
        jobs.append({**j, 'status': 'queued' if affected else 'retained', 'output': str(output),
            'recovery_required': affected, 'original_output': str(old), 'original_request_audit': audit,
            'original_numerical_calls': c['counters']['expensive_evaluations'],
            'original_completed_sha256': digest(old / 'completed.json'),
            'command': replace_output(j['command'], output)})
    if not any(j['recovery_required'] for j in jobs):
        raise ValueError('No balance-affected attempts to recover')
    root.mkdir(exist_ok=False)
    for sub in ('runs', 'logs', 'analysis'):
        (root / sub).mkdir()
    recovery = {**plan, 'id': root.name, 'created': time.time(), 'original_root': str(original),
        'original_plan_sha256': digest(original / 'plan.json'), 'concurrency': concurrency, 'jobs': jobs,
        'recovery_controller_sha256': digest(Path(__file__)),
        'recovery_policy': 'Replace every balance-affected arm-seed from epoch zero; retain all unaffected attempts. No best-attempt selection. No automatic retries.',
        'provider_failure_policy': 'Check balance before dispatch and every minute. Any insufficient-balance response stops active recovery attempts and holds dispatch; all artifacts remain preserved.',
        'max_role_invocations_recovery': sum(j['recovery_required'] for j in jobs) * 40}
    atomic_json(root / 'plan.json', recovery)
    (root / 'plan.sha256').write_text(digest(root / 'plan.json') + '\n')
    shutil.copy2(Path(__file__), root / 'recovery_controller.py')
    atomic_json(root / 'progress.json', {'status': 'prepared', 'jobs': jobs, 'updated': time.time(),
        'active': 0, 'retained': sum(not j['recovery_required'] for j in jobs),
        'completed_replacements': 0, 'queued': sum(j['recovery_required'] for j in jobs)})
    atomic_json(original / 'balance_recovery.json', {'recovery_root': str(root),
        'affected_records': [{'arm': j['arm'], 'seed': j['seed']} for j in jobs if j['recovery_required']],
        'warning': 'Original aggregate includes provider-balance-affected endpoints; do not interpret as a clean 20-seed directional ablation. Original artifacts are preserved.'})
    analyze(root)
    return recovery


def analyze(root):
    from RL4CRN.search_exploration.engine import atomic_json
    from comparisons.search_exploration.analyze_rpa_directional_streams import (
        CONTROL_ROOT, designs, direct_records, pooled_control, topology_union)
    plan = read(root / 'plan.json')
    old_plan = read(CONTROL_ROOT / 'active_plan.json')
    controls = {j['seed']: Path(j.get('output', CONTROL_ROOT / 'runs' / j['name']))
                for j in old_plan['jobs'] if j['arm'] == 'julia'}
    endpoints, unions, audits, errors = [], [], [], []
    for j in plan['jobs']:
        folder = Path(j['output'])
        audit = request_audit(folder)
        audits.append({'arm': j['arm'], 'seed': j['seed'], 'output': str(folder), **audit})
        c = read(folder / 'completed.json')
        if not c or c.get('status') != 'complete' or audit['balance_affected']:
            continue
        try:
            m = read(folder / 'manifest.json')
            if m['communication_mode'] != j['arm'] or m['config']['seed'] != j['seed']:
                raise ValueError('Endpoint identity mismatch')
            row = topology_union(designs(folder), direct_records(folder), m['config']['quality_threshold'])
            endpoints.append({'arm': j['arm'], 'seed': j['seed'], **row})
            if j['arm'] == 'llm-to-rl':
                unions.append(pooled_control(controls[j['seed']], folder))
        except (OSError, ValueError, KeyError) as error:
            errors.append({'arm': j['arm'], 'seed': j['seed'], 'error': str(error)})
    matched = sorted(set(x['seed'] for x in endpoints if x['arm']=='llm-to-rl') &
                     set(x['seed'] for x in endpoints if x['arm']=='rl-to-llm') &
                     set(x['seed'] for x in unions))
    reference = read(FULL_ROOT / 'analysis-vs-adagrad-20260917/summary.json')
    groups = {a: [x for x in endpoints if x['arm']==a and x['seed'] in matched]
              for a in ('llm-to-rl', 'rl-to-llm')}
    groups['independent-union'] = [x for x in unions if x['seed'] in matched]
    for a in ('flash', 'control'):
        groups[a] = [{'seed': x['seed'], 'union_best_loss': x['loss'], 'union_qualified': x['qualified_discovered']}
                     for x in reference['endpoints'] if x['arm']==a and x['seed'] in matched]
    means = {a: {'n': len(rows), 'mean_qualified': float(np.mean([x['union_qualified'] for x in rows])),
                 'median_best_loss': float(np.median([x['union_best_loss'] for x in rows]))}
             for a, rows in groups.items() if rows}
    result = {'completed_valid_endpoints': len(endpoints), 'matched_seeds': matched, 'matched_summary': means,
        'endpoints': endpoints, 'independent_unions': unions, 'request_audits': audits, 'audit_errors': errors,
        'original_balance_affected_records': sum(j['recovery_required'] for j in plan['jobs']),
        'discarded_attempt_numerical_calls': sum(j['original_numerical_calls'] for j in plan['jobs'] if j['recovery_required']),
        'interpretation': 'Partial matched subset; descriptive only. Independent union is post hoc, with a reused insertion-only proposal bank and unmatched schedules.'}
    atomic_json(root / 'analysis/summary.json', result)
    lines = ['# RPA Directional Balance Recovery', '',
        f"Usable directional endpoints: {len(endpoints)}/40. Matched seeds: {matched}.", '',
        '| Workflow | Matched n | Mean qualifying topologies | Median best loss |',
        '|---|---:|---:|---:|']
    for arm in ('control', 'flash', 'llm-to-rl', 'rl-to-llm', 'independent-union'):
        if arm in means:
            v = means[arm]
            lines.append(f"| {arm} | {v['n']} | {v['mean_qualified']:.3f} | {v['median_best_loss']:.7g} |")
    lines += ['', 'Partial results are descriptive, not interim significance tests. Neither losses nor topology counts show a clear full-duplex advantage on the initial 12 unaffected matched seeds. This does not establish equivalence.', '',
        'Independent union combines historical no-LLM search with insertion-only direct supplied-rate LLM proposals. It is not a matched asynchronous no-communication experiment or fixed-proposal causal replay. The proposal-bank reuse creates dependence. Both one-way endpoints include terminal aggregation.', '',
        'Original attempts and costs are preserved. New attempts restart from epoch zero without selecting the better endpoint. Only provider balance failures trigger replacement; ordinary invalid/empty proposals remain experimental outcomes.', '',
        f"Discarded original attempts consumed {result['discarded_attempt_numerical_calls']} numerical calls, additional to replacement budgets. Inference overhead is also additional; it is not estimated by numerical-call counts.", '',
        f'Audit errors: {len(errors)}. See summary.json for per-attempt paths and request audits.', '']
    (root / 'analysis/REPORT.md').write_text('\n'.join(lines))
    return result


def execute(root):
    import fcntl
    from RL4CRN.search_exploration.engine import atomic_json
    plan = read(root / 'plan.json')
    if digest(root / 'plan.json') != (root / 'plan.sha256').read_text().strip():
        raise ValueError('Recovery plan changed')
    if digest(Path(__file__)) != plan['recovery_controller_sha256']:
        raise ValueError('Recovery controller changed')
    verify_inputs(plan)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'started.json').exists():
        raise ValueError('Already started; no automatic retry')
    balance = balance_status()
    atomic_json(root / 'balance.json', balance)
    if not balance['is_available']:
        raise RuntimeError('Insufficient provider balance; nothing launched')
    if external_active(ORIGINAL) or external_active(FULL_ROOT):
        raise RuntimeError('Previous cohort still active')
    atomic_json(root / 'started.json', {'pid': os.getpid(), 'time': time.time()})
    jobs, active, hold = plan['jobs'], {}, False
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'JULIA_NUM_THREADS': '1', 'MPLCONFIGDIR': '/local0/tmp/mpl'}
    env.pop('COMET_API_KEY', None)
    next_balance = time.monotonic() + 60
    balance_verified = True
    while True:
        dirty, outage = False, False
        for name, (process, log, job) in list(active.items()):
            audit = request_audit(Path(job['output']))
            outage = outage or audit['balance_affected']
            if process.poll() is None:
                continue
            log.close()
            completed = read(Path(job['output']) / 'completed.json')
            ok = process.returncode == 0 and completed and completed.get('status')=='complete' and not audit['balance_affected']
            job.update(status='complete' if ok else 'failed', returncode=process.returncode,
                       finished=time.time(), request_audit=audit)
            hold = hold or not ok
            del active[name]
            dirty = True
        if time.monotonic() >= next_balance:
            try:
                balance = balance_status()
                balance_verified = True
                atomic_json(root / 'balance.json', balance)
                outage = outage or not balance['is_available']
            except Exception as error:
                balance_verified = False
                atomic_json(root / 'balance_check_error.json', {'type': type(error).__name__, 'time': time.time()})
            next_balance = time.monotonic() + 60
        if outage:
            hold = True
            atomic_json(root / 'provider_outage.json', {'reason': 'insufficient balance', 'time': time.time()})
            for process, _, job in active.values():
                if process.poll() is None and not job.get('outage_stop_requested'):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    job['outage_stop_requested'] = True
        hold = hold or (root / 'HOLD').exists()
        for job in jobs:
            if hold or not balance_verified or len(active) >= plan['concurrency']:
                break
            if job['status'] != 'queued':
                continue
            try:
                verify_inputs(plan)
            except ValueError as error:
                hold = True
                atomic_json(root / 'dispatch_blocked.json', {'reason': str(error), 'time': time.time()})
                break
            log = (root / 'logs' / (job['name'] + '.log')).open('x')
            process = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status='running', pid=process.pid, started=time.time())
            active[job['name']] = (process, log, job)
            print('Started', job['name'], 'pid', process.pid, flush=True)
        snapshots = [{**j, 'run_status': read(Path(j['output']) / 'status.json') or {}} for j in jobs]
        counts = collections.Counter(j['status'] for j in jobs)
        status = 'held' if hold else 'running' if active else 'waiting_for_balance' if counts['queued'] else 'complete'
        atomic_json(root / 'progress.json', {'status': status, 'updated': time.time(), 'jobs': snapshots,
            'active': len(active), 'retained': counts['retained'], 'completed_replacements': counts['complete'],
            'completed': counts['complete']+counts['retained'], 'failed': counts['failed'], 'queued': counts['queued']})
        if dirty or not active:
            try:
                analyze(root)
            except Exception as error:
                atomic_json(root / 'analysis_failed.json', {'error': str(error), 'time': time.time()})
        if not active and (hold or not counts['queued']):
            return
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.original, args.root)
    if args.run:
        execute(args.root)
    if args.analyze:
        analyze(args.root)
