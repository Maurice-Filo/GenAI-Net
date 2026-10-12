"""Prepare a separate 60-minute RPA cohort from the completed frozen Qwen run.

No scientific source, prompt, per-seed configuration or original output changes.
Launch with the copied resume_local_qwen_controller.py after --check succeeds.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import requests


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(previous, root):
    if root.exists():
        raise FileExistsError(root)
    plan = read(previous / 'plan.json')
    for name, digest in read(previous / 'input_hashes.json').items():
        assert sha(previous / name) == digest, name
    audit = read(previous / 'reference_audit.json')['rpa']
    for name, digest in audit['source_hashes'].items():
        assert sha(previous / 'rpa/source' / name) == digest, name
    original_jobs = [j for j in plan['jobs'] if j['task'] == 'rpa']
    assert sorted(j['seed'] for j in original_jobs) == list(range(1, 21))
    for job in original_jobs:
        assert read(Path(job['output']) / 'completed.json')['status'] == 'complete'
        assert float(job['command'][job['command'].index('--llm-timeout') + 1]) == 900
    environment = read(previous / 'environment.json')
    for name, digest in environment['julia_project_files'].items():
        assert sha(Path(name)) == digest, name
    packages = {name: importlib.metadata.version(name) for name in environment['packages']}
    assert packages == environment['packages'], packages
    session = requests.Session()
    session.trust_env = False
    response = session.get(plan['endpoint'] + '/models', timeout=10)
    response.raise_for_status()
    models = response.json()
    assert any(m['id'] == plan['model'] and m.get('max_model_len') == 262144
               for m in models['data'])
    root.mkdir(parents=True)
    for name in ('logs', 'tmp', 'mpl', 'controller', 'rpa/runs'):
        (root / name).mkdir(parents=True)
    shutil.copytree(previous / 'rpa/source', root / 'rpa/source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(previous / 'rpa/configs', root / 'rpa/configs')
    for path in (previous / 'controller').glob('*.py'):
        shutil.copy2(path, root / 'controller' / path.name)
    shutil.copy2(__file__, root / 'controller' / Path(__file__).name)
    jobs, matches = [], []
    for original in original_jobs:
        job = copy.deepcopy(original)
        job['output'] = original['output'].replace(str(previous), str(root), 1)
        job['command'] = [v.replace(str(previous), str(root), 1) for v in original['command']]
        index = job['command'].index('--llm-timeout') + 1
        job['command'][index] = '3600.0'
        restored = [v.replace(str(root), str(previous), 1) for v in job['command']]
        restored[index] = original['command'][index]
        assert restored == original['command']
        job['timeout_reference'] = original['output']
        jobs.append(job)
        original_manifest = read(Path(original['output']) / 'manifest.json')
        checks = {}
        for kind, key in [('task', 'task_config'), ('exploration', 'config'), ('recipe', 'julia_recipe')]:
            path = root / 'rpa/configs' / f'{kind}-{job["seed"]:02d}.json'
            checks[key] = read(path) == original_manifest[key]
        prompt = Path(job['command'][job['command'].index('--prompt') + 1])
        checks['task_prompt'] = sha(prompt) == original_manifest['task_prompt_sha256']
        assert all(checks.values()), checks
        matches.append(dict(seed=job['seed'], reference=original['output'], matches=checks,
                            only_command_change={'llm_timeout': [900, 3600]}))
    plan.update(jobs=jobs, max_parallel=8, comparison_reference=str(previous),
                differences=['RPA role timeout 900 -> 3600 seconds',
                             'Eight concurrent RPA runs from the start; prior cohort began at two then increased to eight and mixed RPA/Logic',
                             'Separate output paths; all scientific inputs and frozen code unchanged'])
    write(root / 'plan.json', plan)
    write(root / 'reference_audit.json', {'rpa': audit})
    write(root / 'configuration_matches.json', matches)
    write(root / 'execution_settings.json', dict(max_parallel=8, authorization=
          'User requested RPA rerun with 60-minute role timeout; previous authorization for eight concurrent runs'))
    write(root / 'environment.json', dict(time=time.time(), server_models=models, packages=packages,
          julia_project_files=environment['julia_project_files'], reference_environment=str(previous / 'environment.json')))
    write(root / 'comparison_design.json', dict(reference=str(previous), task='rpa', seeds=list(range(1, 21)),
          role_timeout_seconds={'reference': 900, 'new': 3600}, scientific_source_unchanged=True,
          intended_comparisons=['Best loss at common charged allowance', 'Wall-clock convergence',
                               'Successful and failed proposal batches', 'Archive admissions during training vs terminal drain',
                               'Qualifying topology diversity and provenance'],
          caveats=['Asynchronous scheduling and model serving load can also affect arrival times',
                   'Longer timeout does not guarantee more completed proposals',
                   'Context/output token settings are unchanged']))
    frozen = [p for p in root.glob('*.json')]
    frozen += list((root / 'controller').glob('*.py'))
    frozen += list((root / 'rpa/configs').glob('*.json'))
    write(root / 'input_hashes.json', {str(p.relative_to(root)): sha(p) for p in frozen})
    write(root / 'progress.json', dict(updated=time.time(), max_parallel=8, guard=None, jobs=jobs, issues=[]))
    print(json.dumps({'prepared': str(root), 'jobs': len(jobs), 'timeout_seconds': 3600, 'max_parallel': 8}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous.resolve(), args.root.resolve())
