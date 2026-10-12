"""Freeze the approved mechanism-unspecified RPA prompt against the Flash cohort."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

REPO = Path(__file__).resolve().parents[2]
REFERENCE = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def extract(path, destination):
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not (destination / member.name).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive member')
        archive.extractall(destination, filter='data')


def prepare(staging, final):
    staging.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'configs', 'controller', 'logs', 'runs', 'tmp', 'mpl'):
        (staging / name).mkdir()
    source = staging / 'source'
    baseline = REPO / 'comparisons/search_exploration/baseline'
    extract(baseline / 'scientific_source.tar.gz', source)
    support = source / 'comparisons/search_exploration'
    support.mkdir(parents=True, exist_ok=True)
    for path in (REPO / 'comparisons/search_exploration').glob('*.py'):
        shutil.copy2(path, support / path.name)
    for name in ('comparisons/__init__.py', 'comparisons/rpa_search/__init__.py'):
        shutil.copy2(REPO / name, source / name)
    shutil.copytree(baseline, support / 'baseline')
    first = REFERENCE / 'runs/flash-adagrad-rpa-seed01'
    manifest = read(first / 'manifest.json')
    extract(first / 'source.tar.gz', source)
    for name, digest in manifest['new_source'].items():
        assert sha(source / name) == digest, name
    prompt_name = 'comparisons/search_exploration/prompts/rpa-evidence-exploitation-v1.md'
    prompt = source / prompt_name
    original = prompt.read_text()
    assert sha(prompt) == manifest['task_prompt_sha256']
    edits = [
        ('TASK: Evidence-guided design of RPA reaction networks',
         'TASK: Evidence-guided design of reaction networks'),
        ('Use observed limitations to motivate rate changes, reaction replacements\n'
         'or compatible mechanism combinations. Specify each resulting CRN fully.',
         'Use observed limitations to motivate rate changes or changes to the reaction\n'
         'structure. Specify each resulting CRN fully.'),
        ('Consider feedback, integral-like, buffering or other admissible mechanisms\n'
         'without assuming any mechanism class must be best. Avoid optimizing only\n'
         'one disturbance condition.',
         'Avoid optimizing only one disturbance condition.'),
    ]
    revised = original
    for old, new in edits:
        assert revised.count(old) == 1, old
        revised = revised.replace(old, new)
    prompt.write_text(revised)
    (staging / 'TASK.md').write_text(revised)
    (staging / 'reference_TASK.md').write_text(original)
    (staging / 'prompt.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), revised.splitlines(True), fromfile='reference', tofile='mechanism-unspecified')))
    matches, jobs = [], []
    for seed in range(1, 21):
        reference = REFERENCE / 'runs' / f'flash-adagrad-rpa-seed{seed:02d}'
        m = read(reference / 'manifest.json')
        assert read(reference / 'completed.json')['status'] == 'complete'
        for key in ('new_source', 'task_prompt_sha256', 'system_prompt_sha256', 'role_prompts'):
            assert m[key] == manifest[key], (seed, key)
        for kind, key in [('task', 'task_config'), ('exploration', 'config'), ('recipe', 'julia_recipe')]:
            write(staging / 'configs' / f'{kind}-{seed:02d}.json', m[key])
        name = f'rpa-flash-mechanism-unspecified-seed{seed:02d}'
        output = final / 'runs' / name
        command = [str(REPO / '.venv/bin/python'), '-m', 'comparisons.search_exploration.run_harness_streams',
                   '--task-config', str(final / 'configs' / f'task-{seed:02d}.json'),
                   '--exploration-config', str(final / 'configs' / f'exploration-{seed:02d}.json'),
                   '--julia-recipe', str(final / 'configs' / f'recipe-{seed:02d}.json'),
                   '--output', str(output), '--julia-project', '/local0/rossin/git/SynthEvo/SynthEvo',
                   '--dsh-home', '/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home',
                   '--prompt', str(final / 'source' / prompt_name)]
        for key in ('search_workers', 'optimization_workers', 'llm_every', 'llm_candidates', 'llm_timeout'):
            command += ['--' + key.replace('_', '-'), str(m[key])]
        jobs.append(dict(name=name, seed=seed, status='queued', output=str(output), command=command,
                         reference=str(reference), reference_manifest_sha256=sha(reference / 'manifest.json')))
        matches.append(dict(seed=seed, numerical_config_unchanged=True, task_config_unchanged=True,
                            rate_fitting_unchanged=True, shared_and_role_prompts_unchanged=True))
    for name in ('run_rpa_prompt_ablation.py', 'watch_completed_storage.py', 'downsample_closed_diagnostics.py'):
        shutil.copy2(REPO / 'comparisons/search_exploration' / name, staging / 'controller' / name)
    shutil.copy2(__file__, staging / 'controller' / Path(__file__).name)
    write(staging / 'plan.json', dict(schema='rpa-mechanism-unspecified-v1', root=str(final),
        model='deepseek-v4-flash', communication='full', max_parallel=8, epochs=100,
        seeds=list(range(1, 21)), jobs=jobs, reference=str(REFERENCE),
        scientific_changes=['Only the three approved task-prompt edits in prompt.diff'],
        execution_changes=['Eight concurrent runs; asynchronous arrival times may differ', 'Comet disabled'],
        storage_policy='Immediately compact each completed run: stride 10 plus endpoint, preserving all records and full traces for qualifying/ever-HOF topologies',
        interpretation='No explicit named mechanism suggestions; not a claim of an unbiased model',
        exploratory_only=True, task_prompt_sha256=sha(prompt),
        reference_task_prompt_sha256=manifest['task_prompt_sha256']))
    write(staging / 'configuration_matches.json', matches)
    write(staging / 'progress.json', dict(status='prepared', jobs=jobs, issues=[]))
    paths = [p for p in staging.rglob('*') if p.is_file() and p.name != 'progress.json']
    write(staging / 'input_hashes.json', {str(p.relative_to(staging)): sha(p) for p in paths})
    print(json.dumps(dict(prepared=str(staging), destination=str(final), jobs=20, parallel=8)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--final', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.staging.resolve(), args.final.resolve())
