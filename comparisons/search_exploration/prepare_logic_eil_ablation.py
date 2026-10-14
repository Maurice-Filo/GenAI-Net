"""Freeze a full-duplex Logic replication with only imitation_weight set to zero."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from comparisons.search_exploration.prepare_rpa_prompt_ablation import extract, read, sha, write

REPO = Path(__file__).resolve().parents[2]
ENDPOINTS = REPO / 'comparisons/search_exploration/reports/logic5-four-arm-statistics-20260924/endpoints.csv'


def prepare(staging, final):
    with ENDPOINTS.open() as stream:
        refs = {int(row['seed']): Path(row['output']) for row in csv.DictReader(stream)
                if row['arm'] == 'full'}
    assert sorted(refs) == list(range(1, 21))
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
    first = read(refs[1] / 'manifest.json')
    extract(refs[1] / 'source.tar.gz', source)
    for name, digest in first['new_source'].items():
        assert sha(source / name) == digest, name
    prompts = [name for name in first['new_source'] if '/prompts/' in name and name.endswith('.md')]
    assert len(prompts) == 1
    prompt = prompts[0]
    assert sha(source / prompt) == first['task_prompt_sha256']
    shutil.copy2(source / prompt, staging / 'TASK.md')
    jobs, matches = [], []
    for seed, reference in sorted(refs.items()):
        m = read(reference / 'manifest.json')
        assert read(reference / 'completed.json')['status'] == 'complete'
        for key in ('new_source', 'task_prompt_sha256', 'system_prompt_sha256', 'role_prompts'):
            assert m[key] == first[key], (seed, key)
        assert m['communication_mode'] == 'full' and m['model'] == 'deepseek-v4-flash'
        config = dict(m['config'])
        assert config['seed'] == seed and config['epochs'] == 100
        assert config['imitation_weight'] == 0.1
        config['imitation_weight'] = 0.0
        differences = {k: [m['config'][k], v] for k, v in config.items() if m['config'][k] != v}
        assert differences == {'imitation_weight': [0.1, 0.0]}
        for kind, value in [('task', m['task_config']), ('exploration', config), ('recipe', m['julia_recipe'])]:
            write(staging / 'configs' / f'{kind}-{seed:02d}.json', value)
        name = f'logic-flash-full-eil0-seed{seed:02d}'
        command = [str(REPO / '.venv/bin/python'), '-m', 'comparisons.search_exploration.run_logic5_four_arm',
                   '--communication-mode', 'full',
                   '--task-config', str(final / 'configs' / f'task-{seed:02d}.json'),
                   '--exploration-config', str(final / 'configs' / f'exploration-{seed:02d}.json'),
                   '--julia-recipe', str(final / 'configs' / f'recipe-{seed:02d}.json'),
                   '--output', str(final / 'runs' / name), '--julia-project', m['julia_project'],
                   '--dsh-home', '/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home',
                   '--prompt', str(final / 'source' / prompt)]
        for key in ('search_workers', 'optimization_workers', 'llm_every', 'llm_candidates', 'llm_timeout'):
            command += ['--' + key.replace('_', '-'), str(m[key])]
        jobs.append(dict(name=name, seed=seed, status='queued', output=str(final / 'runs' / name),
                         command=command, reference=str(reference),
                         reference_manifest_sha256=sha(reference / 'manifest.json')))
        matches.append(dict(seed=seed, config_differences=differences, task_unchanged=True,
                            julia_recipe_unchanged=True, scientific_source_unchanged=True,
                            task_system_and_role_prompts_unchanged=True))
    for name in ('watch_completed_storage.py', 'downsample_closed_diagnostics.py'):
        shutil.copy2(REPO / 'comparisons/search_exploration' / name, staging / 'controller' / name)
    # Reuse the proven hosted controller and immediate post-completion compaction.
    controller = (REPO / 'comparisons/search_exploration/run_rpa_prompt_ablation.py').read_text()
    controller = controller.replace('Run the frozen Flash prompt ablation,', 'Run the frozen Logic EIL ablation,')
    controller = controller.replace('free < 100 * 2**30', 'free < 30 * 2**30')
    controller = controller.replace('Waiting for 100 GiB free on local1', 'Waiting for 30 GiB free on local1')
    (staging / 'controller/run_logic_eil_ablation.py').write_text(controller)
    shutil.copy2(__file__, staging / 'controller' / Path(__file__).name)
    write(staging / 'plan.json', dict(schema='logic-full-duplex-eil-zero-v1', root=str(final),
        model='deepseek-v4-flash', communication='full', max_parallel=8, epochs=100,
        seeds=list(range(1, 21)), jobs=jobs, reference_endpoints=str(ENDPOINTS),
        scientific_changes=['imitation_weight: 0.1 -> 0.0 only'],
        execution_changes=['Eight concurrent runs; asynchronous arrival times may differ', 'Comet disabled'],
        preserved=['Policy-gradient and entropy terms', 'Archive/speciation and elite sampling',
                   'LLM insertion, search feedback, genetic search and AdaGrad',
                   'All candidate, evaluation and provenance records'],
        storage_policy='Compact each completed run immediately: stride 10 plus endpoint; retain all records and full traces for qualifying/ever-HOF topologies',
        minimum_dispatch_free_gib=30, exploratory_only=True,
        task_prompt_sha256=first['task_prompt_sha256'], system_prompt_sha256=first['system_prompt_sha256']))
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
