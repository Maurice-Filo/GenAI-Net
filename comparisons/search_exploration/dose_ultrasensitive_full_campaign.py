"""Approved 20-seed Ultrasensitive campaign; frozen source, full duplex, issue reporting."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('/local1/rossin/search-exploration/rpa-statistical-20seed-20260915/runs/julia-seed20-no-timeouts/manifest.json')
PROMPT = ROOT / 'comparisons/search_exploration/prompts/dose-ultrasensitive-evidence-exploitation-v1.md'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings(seed):
    reference = read(REFERENCE)
    task = read(ROOT / 'comparisons/llm_crn_generation/configs/paper_breadth_100epoch.json')
    task['search'].update(seed=seed, max_added_reactions=6, rate_constant_range=[.001, 100.])
    task['rl4crn'].update(epochs=100, batch_size=1023, deep_layer_size=1024, n_cpus=6)
    config = {**reference['config'], 'seed': seed, 'workers': 6, 'epochs': 100,
              'quality_threshold': .05, 'require_all_rows': False}
    return task, config


def preflight(output):
    import numpy as np
    from RL4CRN.parameter_optimization import JuliaOptimizer
    from RL4CRN.search_exploration.evaluation import build, evaluate_once
    from RL4CRN.search_streams.julia_adapter import task_experiment
    from RL4CRN.search_streams.julia_stream import LOSS_AGREEMENT
    task_config, _ = settings(1)
    evaluator, task, _ = build('dose_ultrasensitive', task_config, .001, 100.)
    assert evaluator.max_added_reactions == 6
    assert evaluator.forbidden_reaction_ids == frozenset({0, 1})
    library, experiment = task_experiment(evaluator, task)
    inputs = np.asarray(experiment.inputs)[:, 0]
    np.testing.assert_allclose(inputs, np.linspace(0, 1, 10), rtol=1e-6)
    target = 2*inputs**8/(.5**8+inputs**8)
    np.testing.assert_allclose(np.asarray(experiment.targets)[:, 0, 0], target, rtol=1e-6)
    ids = [10, 11, 15, 19, 21, 28]
    starts = [[1., .2, 1., .5, .3, 1.], [.8, .16, .8, .4, .24, .8]]
    checks = []
    with JuliaOptimizer(project=read(REFERENCE)['julia_project'], startup_timeout=600) as worker:
        model = worker.prepare(library.select(ids), experiment)
        for rates in starts:
            canonical = evaluate_once(evaluator, {'reaction_ids': ids, 'rates': rates})
            fitted = worker.evaluate(model, rates, timeout=None)
            assert canonical['simulation_complete']
            np.testing.assert_allclose(fitted['loss'], canonical['loss'],
                                       rtol=LOSS_AGREEMENT['warning_rtol'], atol=LOSS_AGREEMENT['warning_atol'])
            checks.append(dict(rates=rates, canonical_loss=canonical['loss'], julia_loss=fitted['loss']))
        fitted = worker.optimize(model, [starts[1]], steps=2, method='adagrad', learning_rate=.1,
                                 bounds=(.001, 100.), step_policy='feasible', evaluation_limit=4,
                                 timeout=None)['results'][0]
        assert fitted.get('best_rates') is not None
        canonical = evaluate_once(evaluator, {'reaction_ids': ids, 'rates': fitted['best_rates']})
        assert canonical['simulation_complete']
        np.testing.assert_allclose(fitted['best_loss'], canonical['loss'],
                                   rtol=LOSS_AGREEMENT['warning_rtol'], atol=LOSS_AGREEMENT['warning_atol'])
    write(output, dict(status='passed', task='dose_ultrasensitive', checks=checks,
                       fitted_loss=canonical['loss'], fitted_julia_loss=fitted['best_loss'],
                       prompt_sha256=sha(PROMPT), reaction_ids=ids,
                       note='Software checks only; no preflight candidates are injected into search.'))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name in ('configs', 'logs', 'runs', 'source', 'mpl'):
        (root/name).mkdir()
    write(root/'prompt_approval.json', dict(status='approved', approved_by='user', date='2026-10-01',
          instruction="good then queue this too. Let's move to Biphasic",
          task='dose_ultrasensitive', prompt_sha256=sha(PROMPT), epochs=100, seeds=list(range(1,21)),
          other_tasks='Biphasic prompt is pending review; not included in this campaign.'))
    shutil.copy2(PROMPT, root/'TASK.md')
    reference = read(REFERENCE)
    write(root/'reference_manifest.json', reference)
    write(root/'configs/julia_recipe.json', {**reference['julia_recipe'], 'loss_disagreement_policy': 'warn'})
    preflight(root/'preflight.json')
    # Freeze scientific Python/Julia sources and task inputs, without credentials or results.
    paths = set()
    for base in ('RL4CRN', 'comparisons/rpa_search/src', 'comparisons/rpa_search/configs'):
        paths.update(p for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.jl','.json','.yaml','.yml'))
    for base in ('comparisons/llm_crn_generation', 'comparisons/search_exploration'):
        paths.update((ROOT/base).glob('*.py'))
    paths.update((ROOT/'comparisons/llm_crn_generation/configs').glob('*.json'))
    paths.update([PROMPT, ROOT/'tools/dsh_bwrap_resolved_tmp.sh', ROOT/'tools/dsh_headless_continuing_runner.mjs'])
    for folder in ('comparisons',):
        paths.update((ROOT/folder).glob('__init__.py'))
    hashes = {}
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        target = root/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = sha(target)
    write(root/'source_hashes.json', hashes)
    jobs = []
    for seed in range(1, 21):
        task, config = settings(seed)
        write(root/f'configs/task-{seed:02}.json', task)
        write(root/f'configs/search-{seed:02}.json', config)
        name = f'dose-ultrasensitive-full-seed{seed:02}'
        command = [str(ROOT/'.venv/bin/python'), '-m', 'comparisons.search_exploration.run_dose_ultrasensitive_full',
                   '--task-config', str(root/f'configs/task-{seed:02}.json'),
                   '--exploration-config', str(root/f'configs/search-{seed:02}.json'),
                   '--output', str(root/'runs'/name), '--julia-project', reference['julia_project'],
                   '--julia-recipe', str(root/'configs/julia_recipe.json'),
                   '--dsh-home', str(Path.home()/'ai-workspaces/deepseek-test/.dsh-home'),
                   '--prompt', str(root/'source'/PROMPT.relative_to(ROOT)),
                   '--search-workers', '6', '--optimization-workers', '2', '--llm-every', '5',
                   '--llm-candidates', '10', '--llm-timeout', '3600', '--communication-mode', 'full']
        jobs.append(dict(name=name, seed=seed, status='queued', command=command))
    plan = dict(task='dose_ultrasensitive', mode='full', epochs=100, seeds=list(range(1,21)),
                batch_size=1023, budget_per_seed=214793, concurrency=4, quality_threshold=.05,
                quality_threshold_note='Existing ExplorationConfig default, frozen before search. An operational loss cutoff, not a dynamical guarantee. All losses retained for later threshold sweeps.',
                failure_policy='Recoverable request failures and canonical/Julia disagreements recorded; canonical scores authoritative. Failed seeds retained; remaining jobs continue. No silent replacements.',
                provenance='Save all archive events, parent hashes, source/origin, supplied/fitted rates and scores, model requests/responses/workspaces, input/output hashes and terminal unions.',
                source_root=str(root/'source'), source_hashes=hashes,
                input_hashes={str(p.relative_to(root)):sha(p) for p in sorted((root/'configs').glob('*.json'))},
                jobs=jobs, created=time.time())
    write(root/'plan.json', plan)
    (root/'plan.sha256').write_text(sha(root/'plan.json')+'\n')
    (root/'README.md').write_text('# Ultrasensitive dose-response: full duplex\n\n20 independent seeds, 100 epochs each. Approved prompt in TASK.md.\n'
        'Source and inputs frozen; no existing campaigns modified. No Comet uploads.\n'
        'Quality cutoff 0.05 is the existing configuration default, fixed before search.\n'
        'Progress: progress.json. Final issues and endpoint overview: REPORT.md.\n'
        'Recoverable errors do not stop other seeds; completed training with no usable LLM proposals is explicitly flagged.\n')
    return plan


def execute(root):
    from comparisons.search_exploration.recover_rpa_directional_balance import request_audit
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(root/'plan.json')
    assert sha(root/'plan.json') == (root/'plan.sha256').read_text().strip()
    for rel, expected in plan['source_hashes'].items():
        assert sha(root/'source'/rel) == expected, rel
    for rel, expected in plan['input_hashes'].items():
        assert sha(root/rel) == expected, rel
    if (root/'started.json').exists():
        raise RuntimeError('Campaign already started; preserve runs and reconcile before resuming.')
    write(root/'started.json', dict(pid=os.getpid(), started=time.time()))
    env = dict(os.environ, PYTHONPATH=str(root/'source'), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               JULIA_NUM_THREADS='1', MPLCONFIGDIR=str(root/'mpl'), COMET_MODE='DISABLED')
    env.pop('COMET_API_KEY', None)
    active, jobs, issues = {}, plan['jobs'], []
    while True:
        for name, (process, log, job) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            folder = root/'runs'/name
            completed = read(folder/'completed.json')
            audit = request_audit(folder)
            job.update(status='complete' if process.returncode == 0 and completed else 'failed',
                       exit_code=process.returncode, finished=time.time(), request_audit=audit)
            if job['status'] != 'complete':
                issues.append(dict(run=name, kind='failed_run', detail=read(folder/'failed.json')))
            if audit['failed_requests'] or not audit['valid_members']:
                issues.append(dict(run=name, kind='llm_request_issues', detail=audit))
            counters = completed.get('counters', {})
            numerical = {k:v for k,v in counters.items() if v and any(s in k for s in ('disagreement', 'failed', 'unknown'))}
            if numerical:
                issues.append(dict(run=name, kind='numerical_diagnostics', detail=numerical))
            del active[name]
        for job in jobs:
            if len(active) >= plan['concurrency']:
                break
            if job['status'] != 'queued':
                continue
            log = (root/'logs'/f"{job['name']}.log").open('x')
            process = subprocess.Popen(job['command'], cwd=root/'source', env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status='running', pid=process.pid, started=time.time())
            active[job['name']] = process, log, job
            print('Started '+job['name'], flush=True)
        counts = Counter(j['status'] for j in jobs)
        snapshots = [{**j, 'run_status':read(root/'runs'/j['name']/'status.json')} for j in jobs]
        write(root/'progress.json', dict(status='running' if active else 'finished', updated=time.time(),
                                        counts=dict(counts), jobs=snapshots))
        write(root/'issues.json', issues)
        if not active and not counts['queued']:
            break
        time.sleep(5)
    lines = ['# Ultrasensitive full-duplex campaign results', '', f"Completed {counts['complete']}/20; failed {counts['failed']}/20.", '',
             '| Seed | Status | Best archive loss | Qualifying topologies | Valid LLM proposals |', '|---|---|---:|---:|---:|']
    for job in jobs:
        c = read(root/'runs'/job['name']/'completed.json')
        m = c.get('metrics', {})
        lines.append(f"| {job['seed']} | {job['status']} | {m.get('Archive/best_loss', 'NA')} | {m.get('Archive/qualified_topologies', 'NA')} | {job.get('request_audit', {}).get('valid_members', 0)} |")
    lines += ['', '## Recorded issues', '', 'See issues.json for all request, run and numerical failures. No failed seeds are silently replaced.', '',
              'All raw provenance remains in runs/: exploration.sqlite, results.sqlite, julia_jobs/, llm_history.json, direct_llm_proposals.jsonl and harness-workspaces/.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'finished.json', dict(finished=time.time(), counts=dict(counts), issues=len(issues)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--after', type=Path, help='Wait for the preceding campaign to finish before dispatch')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root)
    if args.execute:
        if args.after:
            preceding = args.after.resolve()
            waiting_lock = (root/'waiting.lock').open('a')
            fcntl.flock(waiting_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while not (preceding/'finished.json').exists():
                plan = read(root/'plan.json')
                write(root/'progress.json', dict(status='queued_after_hill', updated=time.time(),
                      counts={'queued': 20}, predecessor=str(preceding), jobs=plan['jobs']))
                time.sleep(10)
        execute(root)
