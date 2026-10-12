"""Freeze and run local-Qwen replications of the reported Flash full-duplex cohorts.

Scientific sources and input dictionaries come from the actual reference runs.
Only the model transport and transport metadata are patched in their frozen code.
No existing campaign or working-tree scientific source is modified.
"""
import argparse
import csv
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

REPO = Path('/local0/rossin/git/GenAI-Net')
PYTHON = REPO / '.venv/bin/python'
MODEL = 'Qwen3.8-27B-DFlash2'
PROVIDER = 'local-qwen38-dflash2-matched'
ENDPOINT = 'http://127.0.0.1:18081/v1'
DSH = Path('/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home')
DEFAULT = Path('/local1/rossin/search-exploration/qwen38-matched-rpa-logic-20seed-20261006-r2')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(archive, dest):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or not (dest / member.name).resolve().is_relative_to(dest.resolve()):
                raise ValueError('Unsafe source archive member')
        tar.extractall(dest, filter='data')


def env(root, task):
    out = os.environ.copy()
    out.pop('COMET_API_KEY', None)
    out.update(PYTHONPATH=str(root / task / 'source'), CUDA_VISIBLE_DEVICES='',
               COMET_MODE='DISABLED', MPLCONFIGDIR=str(root / 'mpl'), TMPDIR=str(root / 'tmp'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JULIA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        out[key] = '1'
    return out


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'tmp', 'mpl', 'controller'):
        (root / name).mkdir()
    shutil.copy2(__file__, root / 'controller' / Path(__file__).name)
    for name in ('watch_completed_storage.py', 'downsample_closed_diagnostics.py'):
        shutil.copy2(REPO / 'comparisons/search_exploration' / name, root / 'controller' / name)
    rpa = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1/runs')
    refs = {'rpa': {s: rpa / f'flash-adagrad-rpa-seed{s:02d}' for s in range(1, 21)}, 'logic': {}}
    with (REPO / 'comparisons/search_exploration/reports/logic5-four-arm-statistics-20260924/endpoints.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['arm'] == 'full':
                refs['logic'][int(row['seed'])] = Path(row['output'])
    jobs, audits = [], {}
    for task, runs in refs.items():
        assert sorted(runs) == list(range(1, 21))
        source = root / task / 'source'
        source.mkdir(parents=True)
        (root / task / 'configs').mkdir()
        (root / task / 'runs').mkdir()
        baseline = REPO / 'comparisons/search_exploration/baseline'
        extract(baseline / 'scientific_source.tar.gz', source)
        for name in ('comparisons/__init__.py', 'comparisons/rpa_search/__init__.py'):
            if (REPO / name).exists():
                shutil.copy2(REPO / name, source / name)
        # Launch/support files absent from the original scientific baseline.
        support = source / 'comparisons/search_exploration'
        support.mkdir(parents=True, exist_ok=True)
        for p in (REPO / 'comparisons/search_exploration').glob('*.py'):
            shutil.copy2(p, support / p.name)
        (support / 'baseline').mkdir()
        for name in ('manifest.json', 'scientific_source.tar.gz'):
            shutil.copy2(baseline / name, support / 'baseline' / name)
        ref = runs[1]
        extract(ref / 'source.tar.gz', source)
        manifest = read(ref / 'manifest.json')
        for name, digest in manifest['new_source'].items():
            assert sha(source / name) == digest, name
        prompts = [n for n in manifest['new_source'] if '/prompts/' in n and n.endswith('.md')]
        assert len(prompts) == 1
        prompt = source / prompts[0]
        assert sha(prompt) == manifest['task_prompt_sha256']
        changes = {}
        harness = source / 'RL4CRN/search_streams/harness.py'
        old = harness.read_text()
        new = old.replace('model="deepseek-v4-flash"', f'model="{MODEL}"').replace(
            'model=model, provider="deepseek-official", timeout_seconds=request_timeout,',
            f'model=model, provider="{PROVIDER}", timeout_seconds=request_timeout,\n'
            f'            openai_compatible_base_url="{ENDPOINT}",\n'
            '            local_context_window=262144, local_max_output_tokens=131072,')
        new = new.replace('class StreamHarnessEvaluator',
            'class LocalMatchedHarnessClient(ExplorationHarnessClient):\n'
            '    """Keep the Flash high-reasoning setting in the local provider profile."""\n'
            '    def _profile_patch_text(self):\n'
            '        return super()._profile_patch_text().replace("reasoning: low", "reasoning: high").replace("low: low", "high: high")\n\n\n'
            'class StreamHarnessEvaluator')
        new = new.replace('client = ExplorationHarnessClient(', 'client = LocalMatchedHarnessClient(')
        assert old != new and 'provider="deepseek-official"' not in new
        harness.write_text(new)
        changes[str(harness.relative_to(source))] = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))
        runner_name = 'run_harness_streams.py' if task == 'rpa' else 'run_logic5_four_arm.py'
        runner = support / runner_name
        old = runner.read_text()
        new = old.replace('"model": "deepseek-v4-flash", "provider": "deepseek-official",',
            f'"model": "{MODEL}", "provider": "{PROVIDER}",\n'
            f'            "openai_compatible_base_url": "{ENDPOINT}",\n'
            '            "local_context_window": 262144, "local_max_output_tokens": 131072, "local_reasoning": "high",')
        new = new.replace('provider balance guarded by controller', 'local endpoint checked by controller')
        assert old != new
        runner.write_text(new)
        changes[str(runner.relative_to(source))] = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))
        # Baseline integrity remains enforced by the unchanged reference runner.
        subprocess.run([str(PYTHON), '-m', 'comparisons.search_exploration.freeze_legacy', '--verify'],
                       cwd=source, env=env(root, task), check=True)
        audits[task] = dict(reference_source=str(ref), transport_diff=changes,
            task_prompt_sha256=sha(prompt), reference_system_prompt_sha256=manifest['system_prompt_sha256'],
            source_hashes={str(p.relative_to(source)): sha(p) for p in source.rglob('*')
                           if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'})
        for seed, ref in sorted(runs.items()):
            m = read(ref / 'manifest.json')
            assert read(ref / 'completed.json')['status'] == 'complete'
            for key in ('new_source', 'role_prompts', 'system_prompt_sha256', 'task_prompt_sha256'):
                assert m[key] == manifest[key], (task, seed, key)
            assert m['config']['seed'] == seed
            config = root / task / 'configs'
            for kind, value in [('task', m['task_config']), ('exploration', m['config']), ('recipe', m['julia_recipe'])]:
                write(config / f'{kind}-{seed:02d}.json', value)
            name = f'{task}-qwen-full-seed{seed:02d}'
            output = root / task / 'runs' / name
            command = [str(PYTHON), '-m', 'comparisons.search_exploration.' + runner.stem,
                '--task-config', str(config / f'task-{seed:02d}.json'),
                '--exploration-config', str(config / f'exploration-{seed:02d}.json'),
                '--julia-recipe', str(config / f'recipe-{seed:02d}.json'),
                '--output', str(output), '--julia-project', '/local0/rossin/git/SynthEvo/SynthEvo',
                '--dsh-home', str(DSH), '--prompt', str(prompt)]
            for option in ('search_workers', 'optimization_workers', 'llm_every', 'llm_candidates', 'llm_timeout'):
                command.extend(['--' + option.replace('_', '-'), str(m[option])])
            if task == 'logic':
                command.extend(['--communication-mode', 'full'])
            jobs.append(dict(name=name, task=task, seed=seed, status='queued', output=str(output), command=command,
                reference=str(ref), reference_manifest_sha256=sha(ref / 'manifest.json')))
    jobs.sort(key=lambda j: (j['seed'], j['task'] != 'rpa'))
    write(root / 'reference_audit.json', audits)
    write(root / 'plan.json', dict(model=MODEL, provider=PROVIDER, endpoint=ENDPOINT, max_parallel=2,
        storage_policy='Immediately after process exit: stride 10 plus endpoint for unprotected diagnostics; preserve all candidates and provenance and full diagnostics for qualifying/ever-HOF topologies',
        differences=['LLM/provider/transport', 'Local context 262144 and output cap 131072 (Flash output cap 256000); reasoning high in both. The local cap leaves room for workspace context.',
                     'Two concurrent seeds; asynchronous proposal arrival can differ', 'Comet disabled; local provenance retained'], jobs=jobs))
    frozen = [root / 'plan.json', root / 'reference_audit.json', *(root / 'controller').glob('*.py')]
    frozen.extend(p for p in root.glob('*/configs/*.json'))
    write(root / 'input_hashes.json', {str(p.relative_to(root)): sha(p) for p in frozen})
    print(json.dumps({'prepared': str(root), 'jobs': len(jobs)}))


def preflight(root, task):
    # Run from the task's frozen source tree, with its actual prompt and evaluator.
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.search_streams.harness import LocalMatchedHarnessClient
    from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
    from RL4CRN.llm.workspace_loss_protocol import PROTOCOL
    from comparisons.search_exploration.freeze_legacy import verify
    verify()
    job = next(j for j in read(root / 'plan.json')['jobs'] if j['task'] == task)
    command = job['command']
    prompt = Path(command[command.index('--prompt') + 1]).read_text()
    timeout = float(command[command.index('--llm-timeout') + 1])
    evaluator, _, _ = build(task, read(root / task / 'configs/task-01.json'), .001, 100.)
    output = root / task / 'preflight'
    output.mkdir(exist_ok=False)
    client = LocalMatchedHarnessClient(workspace_root=output / 'workspaces', dsh_home=DSH,
        provider=PROVIDER, model=MODEL, openai_compatible_base_url=ENDPOINT,
        local_context_window=262144, local_max_output_tokens=131072, timeout_seconds=timeout,
        candidate_validation_policy='independent-members', harness_protocol=PROTOCOL)
    graph = HarnessDeciderWriterCRNGraph(client=client, evaluator=evaluator, writer_retry_limit=0,
        max_workspace_evaluations=0, max_literature_searches=0)
    start = time.monotonic()
    result = graph.run_round(task_description=prompt, num_candidates=10, step=0,
        hall_of_fame_iter=None, forbidden_topologies_text='No topology exclusions supplied.',
        sil_feedback_text='Initial request: RL/HOF/EIL information intentionally withheld.',
        jsonl_path=output / 'evaluations.jsonl')
    summary = dict(task=task, accepted=len(result.evaluations), valid=sum(bool(e.valid) for e in result.evaluations),
        losses=[e.loss for e in result.evaluations], response_validation=result.response_validation,
        workspace=str(client.last_workspace.path), calls=client.last_workspace.call_count,
        seconds=time.monotonic()-start)
    write(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if not summary['valid'] or summary['calls'] != 2:
        raise RuntimeError('Local Decider/Writer preflight failed')


def controller(root):
    import requests
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'started.json').exists():
        raise RuntimeError('No automatic restart; reconcile existing campaign first')
    for name, digest in read(root / 'input_hashes.json').items():
        assert sha(root / name) == digest, name
    for task, audit in read(root / 'reference_audit.json').items():
        for name, digest in audit['source_hashes'].items():
            assert sha(root / task / 'source' / name) == digest, (task, name)
        assert read(root / task / 'preflight/summary.json')['valid'] > 0
    plan = read(root / 'plan.json')
    jobs = plan['jobs']
    write(root / 'started.json', dict(pid=os.getpid(), time=time.time()))
    active, issues = {}, []
    session = requests.Session()
    session.trust_env = False
    while any(j['status'] in ('queued', 'running') for j in jobs):
        for name, (process, log, job) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            run = Path(job['output'])
            job.update(exit_code=process.returncode, ended=time.time())
            job['status'] = 'complete' if process.returncode == 0 and (run / 'completed.json').exists() else 'failed'
            if job['status'] == 'complete':
                # Compact before opening another run, preserving scientific records.
                with (root / 'logs' / f'{name}-storage.log').open('a') as storage_log:
                    compact = subprocess.run([str(PYTHON), '-c',
                        'import sys; from watch_completed_storage import worker; sys.exit(worker(sys.argv[1], 2))', str(run)],
                        cwd=root / 'controller', stdout=storage_log, stderr=subprocess.STDOUT)
                job['storage_exit_code'] = compact.returncode
                if compact.returncode:
                    issues.append(dict(run=name, issue='Compaction incomplete; see storage log'))
                history = read(run / 'llm_history.json') if (run / 'llm_history.json').exists() else []
                job['llm_history_entries'] = len(history)
                if any(h.get('error') or h.get('status') in ('error', 'failed') for h in history):
                    issues.append(dict(run=name, issue='LLM request errors recorded; inspect history'))
            else:
                issues.append(dict(run=name, issue='Training failed; original artifacts retained'))
            del active[name]
        guard = None
        if shutil.disk_usage(root).free < 80 * 2**30:
            guard = 'Waiting for 80 GiB free space'
        if len(active) < plan['max_parallel'] and any(j['status']=='queued' for j in jobs) and guard is None:
            try:
                response = session.get(ENDPOINT + '/models', timeout=10)
                response.raise_for_status()
                assert MODEL in [m['id'] for m in response.json()['data']]
            except Exception as exc:
                guard = 'Local endpoint unavailable: ' + str(exc)
            if guard is None:
                for job in jobs:
                    if job['status'] != 'queued' or len(active) >= plan['max_parallel']:
                        continue
                    log = (root / 'logs' / (job['name'] + '.log')).open('a')
                    process = subprocess.Popen(job['command'], cwd=root / job['task'] / 'source',
                        env=env(root, job['task']), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status='running', pid=process.pid, started=time.time())
                    active[job['name']] = (process, log, job)
        write(root / 'progress.json', dict(updated=time.time(), guard=guard, jobs=jobs, issues=issues))
        if active or any(j['status']=='queued' for j in jobs):
            time.sleep(20)
    write(root / 'finished.json', dict(time=time.time(), issues=issues))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'controller'))
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--task', choices=('rpa', 'logic'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'preflight':
        preflight(args.root, args.task)
    else:
        controller(args.root)
