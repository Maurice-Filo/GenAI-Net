"""Offline entry point for the compact FullDuplexCRNSearch release."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
CODE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = CODE_ROOT.parent if (CODE_ROOT.parent/'RL4CRN').is_dir() else CODE_ROOT
PAPER = Path('analysis/biomolecular_full_duplex_rewrite')
WORK_ROOT = None


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def resolve(path):
    """Translate recorded source locations; never fall back to the source disk."""
    p = Path(path)
    if p.is_relative_to(ROOT) or (WORK_ROOT is not None and p.is_relative_to(WORK_ROOT)):
        return p
    aliases = read(ROOT/'manifest/path_map.json')
    for prefix in sorted(aliases, key=len, reverse=True):
        if p == Path(prefix) or p.is_relative_to(prefix):
            return ROOT/'data/runs'/aliases[prefix]/p.relative_to(prefix)
    raise FileNotFoundError(f'No released mapping for {p}')


def rows(folder, name):
    meta = read(folder/'export.json')['tables'][name]
    with gzip.open(folder/(name+'.jsonl.gz'), 'rt') as f:
        for line in f:
            yield dict(zip(meta['columns'], json.loads(line)))


def verify_one(job):
    folder = ROOT/'data/runs'/job['id']
    meta = read(folder/'export.json')
    topology = set()
    qualifying = 0
    best = None
    for name, table in meta['tables'].items():
        count = 0
        h = hashlib.sha256()
        columns = table['columns']
        with gzip.open(folder/(name+'.jsonl.gz'), 'rb') as f:
            for line in f:
                h.update(line)
                record = json.loads(line)
                assert len(record) == len(columns), (job['id'], name)
                count += 1
                if name == 'designs':
                    d = json.loads(record[columns.index('payload')])
                    if d.get('simulation_complete') and d.get('loss') is not None:
                        loss = d['loss']
                        best = loss if best is None else min(best, loss)
                        if meta.get('threshold') is not None and loss < meta['threshold']:
                            qualifying += 1
                            topology.add(tuple(sorted(d['reaction_ids'])))
        assert count == table['rows'], (job['id'], name, count)
        assert h.hexdigest() == table['rows_sha256'], (job['id'], name, 'digest')
    if 'designs' in meta['tables']:
        assert best == meta['best_design_loss'], job['id']
        if meta.get('threshold') is not None:
            assert qualifying == meta['qualifying_design_rows'], job['id']
            assert len(topology) == meta['qualifying_topologies_in_designs'], job['id']
    source_count = 0
    manifest_path = folder/'manifest.json'
    source_path = folder/'source.tar.gz'
    if manifest_path.exists() and source_path.exists():
        expected = read(manifest_path).get('new_source', {})
        with tarfile.open(source_path, 'r:gz') as archive:
            for name, digest in expected.items():
                stream = archive.extractfile(name)
                assert stream is not None, (job['id'], name)
                assert hashlib.sha256(stream.read()).hexdigest() == digest, (job['id'], name)
                source_count += 1
    return dict(run=job['id'], tables=len(meta['tables']), best_loss=best,
                qualifying_topologies=len(topology), frozen_source_files_checked=source_count)


def verify(output):
    from concurrent.futures import ProcessPoolExecutor
    jobs = read(ROOT/"manifest/runs.json")
    checks = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(verify_one, jobs), 1):
            checks.append(result)
            if i % 20 == 0 or i == len(jobs):
                print(f"Verified {i}/{len(jobs)} runs", flush=True)
    inventory = ROOT/'manifest/checksums.json'
    if inventory.exists():
        verify_files(output)
    write(output/'verification.json', dict(runs=checks, all_passed=True,
                                         file_checksums_checked=inventory.exists()))


def verify_files(output):
    inventory = read(ROOT/'manifest/checksums.json')
    for name, value in inventory.items():
        assert sha(ROOT/name) == value, name
    write(output/'file_verification.json', dict(files=len(inventory), all_passed=True))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_script(path, args=()):
    old = sys.argv
    try:
        sys.argv = [str(path), *args]
        runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv = old


def csv_numbers(path):
    with Path(path).open() as f:
        data = list(csv.DictReader(f))
    for row in data:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return data


def guard_source_access():
    """Fail if plotting tries to silently use an original campaign/repository."""
    forbidden = ('/local0/rossin/git/GenAI-Net/', '/local0/rossin/search-exploration/',
                 '/local1/rossin/search-exploration/', '/local0/home/rossin/ai-workspaces/')
    environment = str(Path(sys.prefix).resolve()) + '/'
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = os.fsdecode(args[0])
        if p.startswith(environment):
            return
        if Path(p).is_absolute() and Path(p).is_relative_to(SOURCE_ROOT):
            if Path(p).is_relative_to(SOURCE_ROOT/'paper'):
                raise PermissionError('Manuscript directory is not a reproduction dependency: '+p)
            return
        if p.startswith(forbidden):
            raise PermissionError('Original source access forbidden during reproduction: '+p)
    sys.addaudithook(audit)


def prepare_analysis(output):
    """Combine frozen inputs with the analysis code in this checkout."""
    global WORK_ROOT
    work = output/'work'
    if work.exists():
        shutil.rmtree(work)
    local = CODE_ROOT/'analysis'
    shutil.copytree(ROOT/'analysis', work/'analysis',
                    ignore=shutil.ignore_patterns('*.py', '__pycache__') if local.is_dir() else None)
    if local.is_dir():
        for src in local.rglob('*.py'):
            dst = work/'analysis'/src.relative_to(local)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    WORK_ROOT = work
    return work


def figures(output):
    work = prepare_analysis(output)
    p = work/PAPER
    (p/'figures').mkdir(exist_ok=True)
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(p))
    os.environ['MPLBACKEND'] = 'Agg'
    guard_source_access()
    f = module('proposed_results_figures', p/'proposed_results_figures.py')
    original_read = f.read
    f.read = lambda path, csv_rows=False: original_read(resolve(path), csv_rows=csv_rows)
    f.style()
    for name, kwargs in [('numerical', {}), ('numerical', {'full': True}),
                         ('communication', {}), ('portfolios', {}), ('breadth', {})]:
        print('Rendering', name, kwargs, flush=True)
        getattr(f, name)(None, **kwargs)
    f.rpa_provenance()
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    convergence = module('release_convergence_plot', p/'plot_convergence.py')
    frozen = convergence.from_frozen()
    # The saved appendix figure uses boxed axes; keep this layout explicit.
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 3,
                         'axes.spines.top': True, 'axes.spines.right': True})
    convergence.plot(frozen, 'search', 'Best design in the numerical search archive',
                     'search_archive_convergence')
    e = module('plot_eil_ablation', p/'plot_eil_ablation.py')
    endpoints = csv_numbers(p/'eil_ablation/endpoints.csv')
    curves = csv_numbers(p/'eil_ablation/curves.csv')
    e.plot(endpoints, curves)
    write(output/'eil_statistics_recomputed.json', e.summarize(endpoints))
    for script in ['plot_prompt_ablation.py', 'plot_logic_provenance.py',
                   'parity_performance/plot_qualification.py',
                   'parity_performance/plot_reactions.py',
                   'qwen_comparison/plot_timeout_supplement.py', 'token_costs/plot_resources.py']:
        print('Rendering', script, flush=True)
        if script == 'plot_logic_provenance.py':
            plt.rcdefaults()
        run_script(p/script)
    expected = read(ROOT/'manifest/figures.json')
    report = []
    for name in expected:
        src = ROOT/'expected/figures'/name
        generated = p/name
        static = name == 'figures/full_duplex_architecture.pdf'
        if static:
            generated.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, generated)
        assert generated.exists(), name
        dest = output/'figures'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, dest)
        record = dict(figure=name, static_diagram=static)
        if shutil.which('pdftoppm'):
            import numpy as np
            from PIL import Image
            with tempfile.TemporaryDirectory() as tmp:
                arrays = []
                for tag, pdf in [('reference', src), ('generated', generated)]:
                    prefix = str(Path(tmp)/tag)
                    subprocess.run(['pdftoppm', '-r', '100', '-png', '-singlefile',
                                    str(pdf), prefix], check=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
                    arrays.append(np.asarray(Image.open(prefix+'.png').convert('RGB')))
                a, b = arrays
                record['same_dimensions'] = a.shape == b.shape
                record['pixel_equal'] = bool(a.shape == b.shape and np.array_equal(a, b))
                if a.shape == b.shape:
                    record['mean_pixel_difference'] = float(np.abs(a.astype(float)-b).mean())
        report.append(record)
    write(output/'figure_report.json', report)
    print(f'Produced {len(report)} figure PDFs in {output / "figures"}', flush=True)


def materialize(run, destination):
    """Recreate the compact SQLite schema for existing diagnostic programs."""
    folder = ROOT/'data/runs'/run
    meta = read(folder/'export.json')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as db:
        for name, table in meta['tables'].items():
            db.execute(table['schema'])
            placeholders = ','.join('?' for _ in table['columns'])
            with gzip.open(folder/(name+'.jsonl.gz'), 'rt') as stream:
                import base64
                def values():
                    for line in stream:
                        yield [base64.b64decode(v['base64']) if isinstance(v, dict) and 'base64' in v else v
                               for v in json.loads(line)]
                db.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', values())


def prepare_run(run, destination):
    """Prepare a fresh experiment directory; never execute numerical/model calls."""
    import shlex
    folder = ROOT/'data/runs'/run
    manifest = read(folder/'manifest.json')
    if manifest.get('task') not in ['rpa', 'logic'] or manifest.get('model') != 'deepseek-v4-flash':
        raise ValueError('The tested preparation command supports Flash RPA/Logic runs; other configurations remain in data/campaigns/')
    if not manifest.get('new_source') or not (folder/'source.tar.gz').exists():
        raise ValueError('This legacy run lacks a complete saved source overlay')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ['RL4CRN', 'comparisons', 'tools', 'numerical_runtime']:
        shutil.copytree(ROOT/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'setup.py', destination/'setup.py')
    def unpack(path):
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive.getmembers():
                target = destination/member.name
                if member.issym() or member.islnk() or not target.resolve().is_relative_to(destination):
                    raise ValueError('Unsafe source archive member: '+member.name)
            archive.extractall(destination)
    unpack(ROOT/'comparisons/search_exploration/baseline/scientific_source.tar.gz')
    unpack(folder/'source.tar.gz')
    for name, expected in manifest['new_source'].items():
        assert sha(destination/name) == expected, name
    config = destination/'replay_config'
    write(config/'task.json', manifest['task_config'])
    write(config/'search.json', manifest['config'])
    write(config/'recipe.json', manifest['julia_recipe'])
    write(config/'reference_manifest.json', manifest)
    prompts = [name for name, digest in manifest['new_source'].items()
               if digest == manifest['task_prompt_sha256']]
    assert len(prompts) == 1, prompts
    runners = [name for name in manifest['new_source'] if
               name.startswith('comparisons/search_exploration/run_') and name.endswith('.py')]
    assert len(runners) == 1, runners
    module_name = runners[0][:-3].replace('/', '.')
    command = ['-m', module_name, '--task-config', 'replay_config/task.json',
               '--exploration-config', 'replay_config/search.json', '--julia-recipe', 'replay_config/recipe.json',
               '--julia-project', 'numerical_runtime/SynthEvo', '--output', 'new_run', '--prompt', prompts[0]]
    if '"--communication-mode"' in (destination/runners[0]).read_text():
        command += ['--communication-mode', manifest['communication_mode']]
    for field, flag in [('search_workers', '--search-workers'), ('optimization_workers', '--optimization-workers'),
                        ('llm_every', '--llm-every'), ('llm_candidates', '--llm-candidates'), ('llm_timeout', '--llm-timeout')]:
        if field in manifest:
            command += [flag, str(manifest[field])]
    script = ('#!/usr/bin/env bash\nset -euo pipefail\n'
              'cd -- "$(dirname -- "${BASH_SOURCE[0]}")"\n'
              ': "${DSH_HOME:?Set DSH_HOME to your configured harness directory}"\n'
              'export MPLCONFIGDIR="$PWD/.matplotlib"\n'
              'exec python '+shlex.join(command)+' --dsh-home "$DSH_HOME" "$@"\n')
    (destination/'run.sh').write_text(script)
    write(destination/'replay_preparation.json', dict(reference_run=run, source_files_verified=len(manifest['new_source']),
          scientific_settings='Copied from reference manifest', automatically_launched=False,
          required='Full numerical dependencies, Julia, configured harness/LLM endpoint'))
    print('Prepared', destination, '— no search or LLM request launched.')


def main():
    global ROOT
    # Scientific checks import these helpers; share the configured data root.
    sys.modules['reproduce'] = sys.modules[__name__]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['verify', 'verify-files', 'analyze', 'figures', 'all', 'materialize', 'prepare-run'])
    ap.add_argument('--data-root', type=Path, default=ROOT,
                    help='Extracted compact release containing data/, analysis/ and manifest/.')
    ap.add_argument('--output', type=Path, default=CODE_ROOT/'output')
    ap.add_argument('--run')
    args = ap.parse_args()
    ROOT = args.data_root.resolve()
    if not (ROOT/'manifest/runs.json').is_file():
        ap.error('Provide --data-root pointing to the extracted compact data release.')
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT.is_relative_to(args.output):
        ap.error('--output must not contain the input release.')
    if args.command in ['materialize', 'prepare-run']:
        if not args.run:
            ap.error('--run is required')
        (materialize if args.command == 'materialize' else prepare_run)(args.run, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'verify-files':
        verify_files(args.output)
    if args.command in ['verify', 'all']:
        verify(args.output)
    if args.command in ['analyze', 'all']:
        from scientific_checks import main as scientific_checks
        analysis = prepare_analysis(args.output)
        scientific_checks(ROOT, args.output, analysis_root=analysis/'analysis', source_root=SOURCE_ROOT)
    if args.command in ['figures', 'all']:
        figures(args.output)


if __name__ == '__main__':
    main()
