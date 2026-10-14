#!/usr/bin/env python3
"""Prepare, check, run and analyse the paper's search experiments.

Preparation and listing require only Python's standard library. Running needs
the full numerical environment. No search or API request occurs before `run`.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESOURCE = 'LLM resource accounting'
# selection, cohort group, short description. Only executed cohorts are entries.
EXPERIMENTS = {
    'rpa-full': (RESOURCE, 'rpa_full', 'RPA, full duplex'),
    'logic-full': (RESOURCE, 'logic_full', 'Logic, full duplex'),
    'hill-full': (RESOURCE, 'hill', 'Hill dose response, full duplex'),
    'ultrasensitive-full': (RESOURCE, 'ultrasensitive', 'Ultrasensitivity, full duplex'),
    'biphasic-full': (RESOURCE, 'biphasic', 'Biphasic response, full duplex'),
    'classification-full': (RESOURCE, 'classifier', 'Classification, full duplex'),
    'parity-full': (RESOURCE, 'parity', 'Four-input parity, full duplex'),
    'rpa-insertion': (RESOURCE, 'rpa_llm-to-rl', 'RPA, LLM to search only'),
    'rpa-feedback': (RESOURCE, 'rpa_rl-to-llm', 'RPA, search to LLM only'),
    'logic-insertion': (RESOURCE, 'logic_llm-to-rl', 'Logic, LLM to search only'),
    'logic-feedback': (RESOURCE, 'logic_rl-to-llm', 'Logic, search to LLM only'),
    'logic-independent': (RESOURCE, 'logic_none', 'Logic, both streams active but no communication'),
    'logic-eil-zero': (RESOURCE, 'logic_eil0', 'Logic, full duplex with imitation weight zero'),
    'rpa-prompt-neutral': (RESOURCE, 'rpa_prompt', 'RPA, prompt without mechanism guidance'),
    'rpa-local': (RESOURCE, 'qwen_rpa', 'RPA, local-model reference settings, 15-minute deadline'),
    'rpa-local-60min': (RESOURCE, 'qwen60', 'RPA, local-model reference settings, 60-minute deadline'),
    'logic-local': (RESOURCE, 'qwen_logic', 'Logic, local-model reference settings'),
    'rpa-numerical': ('Numerical optimization', 'rpa/julia', 'RPA, numerical hybrid, no LLM'),
    'rpa-ga': ('Numerical optimization', 'rpa/ga', 'RPA, standalone GA, no LLM'),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seeds(value):
    result = set()
    try:
        for item in value.split(','):
            bounds = [int(n) for n in item.split('-')]
            if len(bounds) == 1:
                result.add(bounds[0])
            elif len(bounds) == 2 and bounds[0] <= bounds[1]:
                result.update(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError()
        if not result or min(result) < 1 or max(result) > 20:
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError('Use reference seeds 1–20, e.g. 1,3-5 or 1-20')
    return sorted(result)


def select_runs(root, experiment):
    selection, group, _ = EXPERIMENTS[experiment]
    result = {}
    with (root / 'manifest/cohorts.csv').open() as f:
        for row in csv.DictReader(f):
            if row['selection'] == selection and row['group'] == group:
                seed = int(row['seed'])
                if seed in result and result[seed] != row['run']:
                    raise ValueError(f'Ambiguous reference for seed {seed}')
                result[seed] = row['run']
    return result


def model_route(name, home):
    """Qualified names need no config parsing; bare IDs use headless defaults.

    Do not read credentials or execute the JavaScript tags in Cordis YAML.
    """
    if '/' in name:
        provider, model = name.split('/', 1)
        if provider and model:
            return provider, model
    if not name:
        raise ValueError('LLM experiments require --model PROVIDER/MODEL (as configured in Harness)')
    try:
        import yaml
    except ImportError:
        raise ValueError('Use PROVIDER/MODEL, or install PyYAML to resolve a bare model ID')
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor('!', lambda loader, tag, node: None)
    matches = set()
    default = None
    for path in [home / 'cordis.yml', home / 'cordis.patch.yml',
                 home / 'profiles/headless/cordis.yml', home / 'profiles/headless/cordis.patch.yml']:
        if not path.exists():
            continue
        value = yaml.load(path.read_text(), Loader=Loader)
        def walk(obj):
            nonlocal default
            if isinstance(obj, list):
                for x in obj:
                    walk(x)
            elif isinstance(obj, dict):
                cfg = obj.get('config') or {}
                if obj.get('id') == 'agent-default-model' and isinstance(cfg, dict):
                    if cfg.get('provider') and cfg.get('model'):
                        default = (str(cfg['provider']), str(cfg['model']))
                if isinstance(cfg, dict):
                    for provider, settings in (cfg.get('providers') or {}).items():
                        for model in settings.get('models', []):
                            if isinstance(model, dict) and name == model.get('id'):
                                matches.add((str(provider), name))
                for key, child in obj.items():
                    if key != 'config':
                        walk(child)
        walk(value)
    if default and default[1] == name:
        matches.add(default)
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely resolve {name!r}; specify PROVIDER/MODEL from the Harness')
    return next(iter(matches))


def unpack(path, destination):
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if (not (member.isfile() or member.isdir()) or
                    not (destination / member.name).resolve().is_relative_to(destination.resolve())):
                raise ValueError(f'Unsafe archive member: {member.name}')
        archive.extractall(destination)


def restore(root, folder, destination, manifest):
    destination.mkdir()
    for name in ['RL4CRN', 'comparisons', 'tools', 'numerical_runtime']:
        shutil.copytree(root / name, destination / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root / 'setup.py', destination / 'setup.py')
    baseline = root / 'comparisons/search_exploration/baseline'
    base = read(baseline / 'manifest.json')
    if digest(baseline / 'scientific_source.tar.gz') != base['archive_sha256']:
        raise ValueError('Baseline archive checksum mismatch')
    unpack(baseline / 'scientific_source.tar.gz', destination)
    unpack(folder / 'source.tar.gz', destination)
    expected = {**base['files'], **manifest['new_source']}
    for name, checksum in expected.items():
        if digest(destination / name) != checksum:
            raise ValueError(f'Reference source mismatch: {name}')
    # Record every executable source, including transport and numerical runtime.
    return {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.jl', '.mjs', '.sh', '.toml', '.json', '.md'}}


def progress(done, total, message, final=False):
    if sys.stderr.isatty():
        filled = int(24 * done / max(1, total))
        print(f'\r[{"#" * filled}{"-" * (24-filled)}] {done}/{total} {message}\033[K',
              end='\n' if final else '', file=sys.stderr, flush=True)
    elif final:
        print(f'{done}/{total} {message}', flush=True)


def prepare(a):
    root = a.data_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    refs = select_runs(root, a.experiment)
    absent = set(a.seeds) - refs.keys()
    if absent:
        raise ValueError(f'Missing reference seeds: {sorted(absent)}')
    manifests = {s: read(root / 'data/runs' / refs[s] / 'manifest.json') for s in a.seeds}
    harness = any(m.get('harness') for m in manifests.values())
    if not harness and any(getattr(a, key) is not None for key in ['llm_every','llm_candidates','llm_timeout']):
        raise ValueError('LLM overrides do not apply to a numerical-only experiment')
    if a.experiment == 'rpa-ga' and (a.epochs is not None or a.optimization_workers is not None):
        raise ValueError('Standalone GA uses --evaluation-budget and --search-workers, not epochs or fitting workers')
    home = a.dsh_home.expanduser().resolve() if a.dsh_home else None
    if harness and not home:
        raise ValueError('Set DSH_HOME or supply --dsh-home; configure the model in that Harness')
    provider, model = model_route(a.model or '', home) if harness else (None, None)
    if not harness and a.model:
        raise ValueError('This numerical-only experiment does not use --model')
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / 'launcher'
    launcher.mkdir()
    for name in ['search_worker.py', 'search_transport.py']:
        shutil.copy2(HERE / name, launcher / name)
    shutil.copy2(REPO / 'comparisons/search_exploration/downsample_closed_diagnostics.py', launcher / 'compact.py')
    campaign = dict(schema='fullduplex-search-campaign-v1', experiment=a.experiment,
                    description=EXPERIMENTS[a.experiment][2], parallel_runs=a.parallel_runs,
                    python=sys.executable, model=model, provider=provider,
                    dsh_home=str(home) if home else None, compaction=not a.no_compact,
                    launcher_hashes={p.name: digest(p) for p in launcher.iterdir()}, jobs=[])
    try:
        for i, seed in enumerate(a.seeds):
            m = manifests[seed]
            folder = root / 'data/runs' / refs[seed]
            jobdir = output / 'jobs' / f'seed-{seed:02d}'
            jobdir.mkdir(parents=True)
            source = jobdir / 'source'
            hashes = restore(root, folder, source, m)
            cfg = dict(m['config'])
            overrides = {}
            for field in ['search_workers', 'optimization_workers', 'llm_every', 'llm_candidates', 'llm_timeout']:
                if getattr(a, field) is not None:
                    overrides[field] = {'reference': m.get(field), 'effective': getattr(a, field)}
            for field in ['epochs', 'batch_size', 'evaluation_budget']:
                value = getattr(a, field)
                if value is not None:
                    if field == 'epochs' and a.experiment == 'rpa-ga':
                        raise ValueError('Standalone GA uses --evaluation-budget, not --epochs')
                    overrides[field] = {'reference': cfg.get(field), 'effective': value}
                    cfg[field] = value
            if cfg.get('epochs', 0) * cfg['batch_size'] > cfg['evaluation_budget']:
                raise ValueError('Evaluation budget must cover epochs × batch size')
            write(jobdir / 'reference_manifest.json', m)
            write(jobdir / 'task.json', m['task_config'])
            write(jobdir / 'search.json', cfg)
            write(jobdir / 'recipe.json', m.get('julia_recipe') or {})
            runners = [n for n in m['new_source'] if n.startswith('comparisons/search_exploration/run_') and n.endswith('.py')]
            if len(runners) != 1:
                raise ValueError(f'Expected one reference runner, found {runners}')
            module = runners[0][:-3].replace('/', '.')
            text = (source / runners[0]).read_text()
            argv = ['--output', str(jobdir / 'output')]
            if a.experiment == 'rpa-ga':
                # Restore the actual GA donor manifest, never fabricate a donor run.
                aliases = read(root / 'manifest/path_map.json')
                donor_id = aliases.get(m['reference'])
                if not donor_id:
                    raise ValueError('GA donor reference missing from compact release')
                donor = jobdir / 'ga_reference'; donor.mkdir()
                for name in ['manifest.json', 'completed.json']:
                    shutil.copy2(root / 'data/runs' / donor_id / name, donor / name)
                argv += ['--reference', str(donor), '--seed', str(seed), '--budget', str(cfg['evaluation_budget']),
                         '--batch-size', str(cfg['batch_size']), '--workers', str(a.search_workers or m['search_workers']), '--no-comet']
                if cfg.get('evaluation_seconds') is None:
                    argv += ['--no-evaluation-timeouts']
            else:
                argv += ['--task-config', str(jobdir / 'task.json'), '--exploration-config', str(jobdir / 'search.json'),
                         '--julia-recipe', str(jobdir / 'recipe.json'), '--julia-project', str(source / 'numerical_runtime/SynthEvo')]
                if '"--task"' in text:
                    argv += ['--task', m['task']]
                for field in ['search_workers', 'optimization_workers', 'queue_capacity', 'optimize_per_epoch', 'llm_every', 'llm_candidates', 'llm_timeout']:
                    flag = '--' + field.replace('_', '-')
                    if '"' + flag + '"' not in text:
                        continue
                    value = getattr(a, field, None)
                    if value is not None:
                        overrides[field] = {'reference': m.get(field), 'effective': value}
                    else:
                        value = m.get(field)
                    if value is not None:
                        argv += [flag, str(value)]
                if '"--embedding-size"' in text:
                    argv += ['--embedding-size', str(m.get('policy', {}).get('deep_layer_size', 1024))]
                if m.get('harness'):
                    prompts = [n for n, h in m['new_source'].items() if h == m['task_prompt_sha256']]
                    if len(prompts) != 1:
                        raise ValueError('Cannot identify the exact reference prompt')
                    argv += ['--prompt', str(source / prompts[0]), '--dsh-home', str(home)]
                    if '"--communication-mode"' in text:
                        argv += ['--communication-mode', m['communication_mode']]
            job = dict(seed=seed, reference_run=refs[seed], experiment=a.experiment, task=m['task'],
                       harness=bool(m.get('harness')), model=model, provider=provider, module=module, argv=argv,
                       overrides=overrides, epochs=cfg.get('epochs'), evaluation_budget=cfg['evaluation_budget'],
                       source_hashes=hashes, input_hashes={str(p.relative_to(jobdir)): digest(p)
                           for p in jobdir.rglob('*.json') if not p.is_relative_to(source)},
                       search_workers=a.search_workers or m.get('search_workers'),
                       optimization_workers=a.optimization_workers or m.get('optimization_workers', 0))
            write(jobdir / 'job.json', job)
            campaign['jobs'].append(dict(seed=seed, path=str(jobdir.relative_to(output)), state='pending',
                                         job_sha256=digest(jobdir / 'job.json')))
            progress(i+1, len(a.seeds), f'prepared seed {seed}', final=not sys.stderr.isatty())
        write(output / 'campaign.json', campaign)
        progress(len(a.seeds), len(a.seeds), 'Prepared; no search or model request launched.', final=True)
        print(f'Campaign: {output}\nNext: python reproduce/search.py check {output}')
    except BaseException as error:
        write(output / 'preparation_failed.json', {'error': str(error)})
        raise


def campaign_path(value):
    return Path(value).expanduser().resolve()


def job_status(root, entry):
    output = root / entry['path'] / 'output'
    status = read(output / 'status.json') if (output / 'status.json').exists() else {}
    done = read(output / 'completed.json') if (output / 'completed.json').exists() else {}
    return status, done


def check(a):
    root = campaign_path(a.campaign); campaign = read(root / 'campaign.json')
    issues = []
    python_environment = julia_environment = None
    for name, expected in campaign['launcher_hashes'].items():
        if digest(root / 'launcher' / name) != expected:
            issues.append(f'Launcher changed: {name}')
    for entry in campaign['jobs']:
        base = root / entry['path']; job = read(base / 'job.json')
        if digest(base / 'job.json') != entry['job_sha256']:
            issues.append(f'Seed {entry["seed"]}: job.json changed')
        for category, prefix in [('source_hashes', base / 'source'), ('input_hashes', base)]:
            for name, expected in job[category].items():
                if not (prefix / name).is_file() or digest(prefix / name) != expected:
                    issues.append(f'Seed {entry["seed"]}: changed/missing {name}')
    if not a.files_only:
        first = root / campaign['jobs'][0]['path'] / 'source'
        env = environment(root, first)
        code = ('import torch, numpy, scipy, gymnasium, sksundae, orjson, json, sys, platform; '
                'from RL4CRN.search_exploration.evaluation import build; '
                'print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(), '
                'torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__, sksundae=sksundae.__version__)))')
        result = subprocess.run([a.python or campaign['python'], '-c', code], cwd=first, env=env, capture_output=True, text=True)
        if result.returncode:
            issues.append('Numerical Python environment: ' + result.stderr[-1200:])
        else:
            python_environment = json.loads(result.stdout.strip().splitlines()[-1])
        needs_julia = campaign['experiment'] != 'rpa-ga'
        if needs_julia:
            if not shutil.which('julia'):
                issues.append('Julia is not on PATH')
            else:
                try:
                    script = ('import json\nfrom RL4CRN.parameter_optimization import JuliaOptimizer\n'
                              'with JuliaOptimizer(project=' + repr(str(first / 'numerical_runtime/SynthEvo')) +
                              ', startup_timeout=160) as worker:\n    print(json.dumps(worker.info))\n')
                    result = subprocess.run([a.python or campaign['python'], '-c', script], cwd=first, env=env,
                                            capture_output=True, text=True, timeout=180)
                    if result.returncode:
                        issues.append('Julia environment: ' + result.stderr[-1200:])
                    else:
                        julia_environment = json.loads(result.stdout.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    issues.append('Julia startup exceeded 180 seconds; instantiate/precompile it first')
        if campaign['model']:
            home = Path(campaign['dsh_home'])
            if not (home / 'profiles/headless/package.json').is_file():
                issues.append('Harness headless profile missing; run tools/install_harness.sh')
            # Check executable discovery using the same command selector as the client.
            command = subprocess.run([a.python or campaign['python'], '-c',
                'import json; from RL4CRN.llm.harness_client import _default_dsh_command; print(json.dumps(_default_dsh_command()))'],
                cwd=first, env=env, capture_output=True, text=True)
            if command.returncode:
                issues.append('Cannot resolve Harness command')
            else:
                cmd = json.loads(command.stdout.strip().splitlines()[-1])
                if not shutil.which(cmd[0]):
                    issues.append(f'Harness executable not found: {cmd[0]}; source the installer env.sh')
            if not shutil.which('bwrap'):
                issues.append('bubblewrap is not installed')
    report = dict(ok=not issues, issues=issues, seeds=len(campaign['jobs']), model=campaign['model'],
                  provider=campaign['provider'], live_model_checked=False, files_only=a.files_only,
                  python_environment=python_environment, julia_environment=julia_environment,
                  free_disk_gib=round(shutil.disk_usage(root).free/2**30, 2))
    write(root / 'check.json', report)
    print(f'Preflight {"FAILED" if issues else "passed"}: {len(campaign["jobs"])} seed(s), '
          f'{report["free_disk_gib"]:g} GiB free. No model request sent.')
    for issue in issues:
        print('  - ' + issue)
    if issues:
        raise ValueError('Preflight failed; see check.json')
    return report


def environment(root, source):
    env = dict(os.environ)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JULIA_NUM_THREADS']:
        env[key] = '1'
    env.update(PYTHONPATH=str(source), MPLCONFIGDIR=str(root / '.matplotlib'),
               COMET_MODE='DISABLED', CHOKIDAR_USEPOLLING='1', CHOKIDAR_INTERVAL='1000')
    return env


def compact(root, entry, python):
    output = root / entry['path'] / 'output'
    with (root / entry['path'] / 'compaction.log').open('a') as log:
        result = subprocess.run([python, str(root / 'launcher/compact.py'), str(output), '--workers', '1'],
                                stdout=log, stderr=subprocess.STDOUT, env=environment(root, root))
    entry['compaction'] = 'complete' if result.returncode == 0 else 'failed-see-log'


def run(a):
    root = campaign_path(a.campaign); campaign = read(root / 'campaign.json')
    python = a.python or campaign['python']
    check(argparse.Namespace(campaign=root, python=python, files_only=False))
    lock = (root / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    parallel = a.parallel_runs or campaign['parallel_runs']
    try:
        for entry in campaign['jobs']:
            base = root / entry['path']
            if entry['state'] == 'running':
                try:
                    os.kill(entry['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f'Seed {entry["seed"]} process {entry["pid"]} may still be alive; inspect it before restarting')
                entry['state'] = 'interrupted'
            if entry['state'] == 'pending' and (base / 'output').exists():
                raise ValueError(f'Existing output for seed {entry["seed"]}; refusing to overwrite')
            if entry['state'] == 'complete' and campaign['compaction'] and entry.get('compaction') != 'complete':
                compact(root, entry, python)
        last_print = 0
        while True:
            for seed, (process, log, entry) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                _, done = job_status(root, entry)
                entry.update(state='complete' if process.returncode == 0 and done.get('status') == 'complete' else 'failed',
                             exit_code=process.returncode, finished=time.time())
                del active[seed]
                if entry['state'] == 'complete' and campaign['compaction']:
                    compact(root, entry, python)
            for entry in campaign['jobs']:
                if len(active) >= parallel:
                    break
                if entry['state'] != 'pending':
                    continue
                base = root / entry['path']
                log = (base / 'run.log').open('a')
                process = subprocess.Popen([python, '-u', str(root / 'launcher/search_worker.py'), str(base / 'job.json')],
                    cwd=base / 'source', env=environment(root, base / 'source'), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                entry.update(state='running', pid=process.pid, started=time.time())
                active[entry['seed']] = (process, log, entry)
            write(root / 'campaign.json', campaign)
            completed = sum(e['state'] == 'complete' for e in campaign['jobs'])
            failed = sum(e['state'] in {'failed', 'interrupted'} for e in campaign['jobs'])
            epochs = sum(job_status(root, e)[0].get('epochs', job_status(root, e)[0].get('epoch', 0)) for e in campaign['jobs'])
            message = f'complete | running {len(active)} | failed {failed} | recorded epochs {epochs}'
            if sys.stderr.isatty() or time.monotonic() - last_print > 30 or not active:
                progress(completed, len(campaign['jobs']), message, final=not active or not sys.stderr.isatty())
                last_print = time.monotonic()
            if not active:
                break
            time.sleep(2)
    except BaseException:
        for process, _, _ in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log, entry in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            log.close(); entry.update(state='interrupted', finished=time.time())
        raise
    finally:
        write(root / 'campaign.json', campaign)
        lock.close()
    if any(e['state'] != 'complete' or e.get('compaction', 'complete') != 'complete' for e in campaign['jobs']):
        raise ValueError('Some runs or compactions need attention; see status and per-seed logs. Attempts preserved.')


def status(a):
    root = campaign_path(a.campaign); c = read(root / 'campaign.json')
    print(f'{c["experiment"]} | model: {c["provider"]}/{c["model"]}' if c['model'] else c['experiment'])
    print(f'{"Seed":>4}  {"State":<13} {"Epochs":>7}  {"Best loss":>12}  Compaction')
    for entry in c['jobs']:
        s, done = job_status(root, entry)
        metrics = done.get('metrics', s.get('metrics', {}))
        loss = metrics.get('Archive/best_loss')
        value = f'{loss:.5g}' if isinstance(loss, (int, float)) else '—'
        print(f'{entry["seed"]:4}  {entry["state"]:<13} {s.get("epochs", s.get("epoch", 0)):7}  {value:>12}  {entry.get("compaction", "—")}')


def analyze(a):
    from search_analysis import analyze_campaign
    analyze_campaign(campaign_path(a.campaign), rpa_screen=a.rpa_screen)


def diagnose(a):
    from search_diagnostics import diagnose as run_diagnostic
    run_diagnostic(campaign_path(a.campaign), a.workers)


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Start: list → prepare → check → run → status/analyze. See reproduce/SEARCH.md.')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List the executed experiment presets')
    q = sub.add_parser('prepare', help='Restore reference settings and sources; no computation or API calls')
    q.add_argument('--experiment', choices=EXPERIMENTS, metavar='NAME', required=True, help='Named preset; see the list command')
    q.add_argument('--data-root', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--seeds', '--seed', type=seeds, default=[1], help='1–20; comma lists/ranges, default 1')
    q.add_argument('--parallel-runs', type=positive, default=1)
    q.add_argument('--model', help='Configured Harness PROVIDER/MODEL, or an unambiguous configured model ID')
    q.add_argument('--dsh-home', type=Path, default=os.environ.get('DSH_HOME'))
    for flag in ['epochs', 'batch-size', 'evaluation-budget', 'search-workers', 'optimization-workers', 'llm-every', 'llm-candidates', 'llm-timeout']:
        q.add_argument('--'+flag, type=positive, help='Override recorded setting; saved explicitly in provenance')
    q.add_argument('--no-compact', action='store_true', help='Opt out of completion-time diagnostic downsampling')
    q = sub.add_parser('check', help='Offline integrity, Python, Julia and Harness installation checks')
    q.add_argument('campaign'); q.add_argument('--python'); q.add_argument('--files-only', action='store_true')
    q = sub.add_parser('run', help='Launch pending seeds; makes model requests for LLM experiments')
    q.add_argument('campaign'); q.add_argument('--python'); q.add_argument('--parallel-runs', type=positive)
    q = sub.add_parser('diagnose', help='Run additional Logic reduction/perturbation simulations')
    q.add_argument('campaign'); q.add_argument('--workers', type=positive, default=2)
    for command, help_text in [('status', 'Show per-seed progress'), ('analyze', 'Plot fresh results and export counts/provenance')]:
        q = sub.add_parser(command, help=help_text); q.add_argument('campaign')
        if command == 'analyze':
            q.add_argument('--rpa-screen', action='store_true', help='Also apply the paper\'s RPA motif/balance screens')
    a = p.parse_args()
    try:
        if a.command == 'list':
            for name, (_, _, description) in EXPERIMENTS.items():
                print(f'{name:25} {description}')
        else:
            globals()[a.command](a)
    except KeyboardInterrupt:
        p.exit(130, '\nInterrupted. Existing attempts were preserved; use status before restarting.\n')
    except (ValueError, OSError, KeyError) as error:
        p.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
