"""Recover manuscript recipes and freeze an isolated, reproducible two-arm campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')

def pilot_julia_recipe(manifest):
    # User-authorized shared amendment: log disagreements and retain canonical scoring.
    return dict(manifest['julia_recipe'], loss_disagreement_policy='warn')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-logic-from', type=Path)
    parser.add_argument('--wait-for-rpa-from', type=Path)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((ROOT/'paper/jmlr_hybrid_crn/generated/channels.json').read_text())['rows']
    selected = [r for r in rows if r['arm'] == 'none' and r['seed'] in (1,2,3)]
    audit = []
    manifests = {}
    for r in selected:
        folder = Path(r['folder']); m = json.loads((folder/'manifest.json').read_text())
        manifests[r['task'],r['seed']] = m
        mismatches = [k for k,v in m['new_source'].items()
                      if not (ROOT/k).is_file() or digest(ROOT/k) != v]
        write(out/'references'/f"{r['task']}-seed{r['seed']}-manifest.json",m)
        write(out/'references'/f"{r['task']}-seed{r['seed']}-completed.json",json.loads((folder/'completed.json').read_text()))
        audit.append({'task':r['task'],'seed':r['seed'],'folder':str(folder),
            'manifest_sha256':digest(folder/'manifest.json'),'source_mismatches':mismatches,
            'reused':False,'reason':'Fresh matched baseline: incompatible old code or isolated LLM activity; learning diagnostics missing'})
    write(out/'historical_control_audit.json',audit)
    jobs=[]
    for task in ('rpa','logic'):
        for seed in (1,2,3):
            m=manifests[task,seed]
            # The current manuscript's RPA seeds 2/3 use the no-timeout completion protocol.
            # Apply that current protocol to seed 1 in BOTH new arms; record the difference.
            cfg=dict(m['config'], evaluation_seconds=None, fit_seconds=None)
            recipe=pilot_julia_recipe(m)
            task_cfg=m['task_config']
            base=out/'configs'/f'{task}-seed{seed}'
            for suffix,value in [('exploration',cfg),('task',task_cfg),('julia',recipe)]:
                write(base.with_name(base.name+'-'+suffix+'.json'),value)
            for condition in ('baseline','ablation'):
                name=f'{task}-{condition}-seed{seed:02d}'
                cmd=[str(ROOT/'.venv/bin/python'),'-m','comparisons.search_exploration.category_pilot.run',
                    '--condition',condition,'--task',task,'--task-config',str(base)+'-task.json',
                    '--exploration-config',str(base)+'-exploration.json','--julia-recipe',str(base)+'-julia.json',
                    '--julia-project',m['julia_project'],'--output',str(out/'runs'/name),
                    '--embedding-size','1024','--search-workers',str(m['search_workers']),
                    '--optimization-workers',str(m['optimization_workers']),
                    '--queue-capacity',str(m['queue_capacity']),'--optimize-per-epoch',str(m['optimize_per_epoch'])]
                job={'name':name,'task':task,'seed':seed,'condition':condition,'command':cmd}
                if task == 'logic' and a.reuse_logic_from:
                    prior=(a.reuse_logic_from/'runs'/name).resolve()
                    prior_manifest=json.loads((prior/'manifest.json').read_text())
                    if (prior_manifest['config'] != cfg or prior_manifest['task_config'] != task_cfg
                        or prior_manifest['julia_recipe'] != recipe or prior_manifest['condition'] != condition
                        or prior_manifest['harness'] or prior_manifest['model'] is not None):
                        raise ValueError('Existing Logic run differs from amended pilot: '+name)
                    for key in ('search_workers','optimization_workers','queue_capacity','optimize_per_epoch'):
                        if prior_manifest[key] != m[key]:raise ValueError('Worker/budget mismatch: '+key)
                    science=[key for key in prior_manifest['new_source'] if key.startswith('RL4CRN/')
                             or key.endswith('/intervention.py') or key.endswith('/run.py')]
                    if any(digest(ROOT/key) != prior_manifest['new_source'][key] for key in science):
                        raise ValueError('Existing Logic numerical/training sources differ')
                    if prior_manifest['legacy_source'] != digest(ROOT/'comparisons/search_exploration/baseline/scientific_source.tar.gz'):
                        raise ValueError('Legacy source mismatch')
                    import importlib.metadata
                    for package, version in prior_manifest['versions'].items():
                        if importlib.metadata.version(package) != version:raise ValueError('Runtime version changed: '+package)
                    if prior_manifest['search_evaluation_reserve'] != cfg['epochs']*cfg['batch_size']:
                        raise ValueError('Search budget mismatch')
                    job.update(reuse_existing=True, reused_live_from=str(prior),
                        reused_manifest_sha256=digest(prior/'manifest.json'))
                jobs.append(job)
    # Snapshot current dirty scientific sources rather than relying on git HEAD.
    snapshot=out/'source'; snapshot.mkdir()
    paths=set()
    for folder in ['RL4CRN','comparisons/llm_crn_generation','comparisons/llm_seeded_evolution',
                   'comparisons/rpa_search/src','comparisons/rpa_search/configs']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.jl','.json','.toml')
            and not any(x in ('__pycache__','reports','campaigns','generated','data') for x in p.relative_to(ROOT).parts))
    paths.update((ROOT/'comparisons/search_exploration').glob('*.py'))
    paths.update((ROOT/'comparisons/search_exploration/category_pilot').glob('*.py'))
    paths.add(ROOT/'comparisons/search_exploration/category_pilot/README.md')
    paths.update((ROOT/'comparisons/search_exploration/baseline').iterdir())
    frozen=json.loads((ROOT/'comparisons/search_exploration/baseline/manifest.json').read_text())
    paths.update(ROOT/k for k in frozen['files'])
    paths.add(ROOT/'tests/test_category_pilot.py')
    for path in list(paths):
        for parent in path.parents:
            if parent == ROOT:break
            if (parent/'__init__.py').is_file():paths.add(parent/'__init__.py')
    hashes={}
    for p in sorted(paths):
        if p.is_file():
            rel=p.relative_to(ROOT); dest=snapshot/rel; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest);hashes[str(rel)]=digest(dest)
    # Every Python child loads this guard from the frozen PYTHONPATH.
    guard = snapshot/'sitecustomize.py'
    guard.write_text("import sys, socket, os, json\ndef guard(event, args):\n    if event == 'socket.connect' and args[0].family in (socket.AF_INET, socket.AF_INET6):\n        path=os.environ.get('PILOT_NETWORK_AUDIT')\n        if path:\n            with open(path, 'a') as f: f.write(json.dumps({'event':event,'address':str(args[1])})+'\\n')\n        raise RuntimeError('External socket connections prohibited in no-LLM pilot')\nsys.addaudithook(guard)\n")
    hashes['sitecustomize.py']=digest(guard)
    write(out/'source_hashes.json',hashes)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for rel in hashes:tar.add(snapshot/rel,arcname=rel,recursive=False)
    julia=Path(manifests['rpa',2]['julia_project'])
    write(out/'julia_source_hashes.json',{str(p):digest(p) for p in sorted(julia.rglob('*'))
          if p.is_file() and (p.suffix in ('.jl','.toml')) and '.git' not in p.parts})
    write(out/'campaign.json',{'schema':'single-category-negative-loss-pilot-v1','jobs':jobs,
        'concurrency':7, 'task_concurrency':{'rpa':1,'logic':6},'source_root':str(snapshot),'source_archive_sha256':digest(out/'source.tar.gz'),
        'historical_controls':False,'epochs':100,'seeds':[1,2,3],
        'protocol_amendment':'User requested recording numerical loss disagreements without aborting. Both conditions use warn; canonical loss stays authoritative. All thresholds/tolerances and budgets unchanged.',
        'wait_for_prior_rpa_exits':[str(f.with_name(f.name.replace('.process.json','.exit.json')).resolve())
            for f in sorted((a.wait_for_rpa_from/'logs').glob('rpa-*.process.json'))] if a.wait_for_rpa_from else [],
        'shared_protocol_changes_from_reference':{'rpa_seed1':'Current no-timeout protocol from manuscript seeds 2/3; fresh baseline also uses current fitting implementation'},
        'candidate_pool':'All complete committed search proposals and canonically reevaluated retained fitting results, all numerical sources; no internal Julia trials',
        'failure_mapping':'R=-canonical loss, including incomplete penalty traces. No normalization/clipping. Missing/nonfinite canonical result aborts without update.',
        'context':'Four components unchanged; category coverage is 0 or 1/2000 in ablation.',
        'scope':'Combined ablation; cannot isolate speciation; no pruning or perturbations'})
    (out/'logs').mkdir(); (out/'runs').mkdir()
    for job in jobs:
        if job.get('reuse_existing'):
            (out/'runs'/job['name']).symlink_to(job['reused_live_from'], target_is_directory=True)
    write(out/'configuration_hashes.json',{str(f.relative_to(out)):digest(f) for f in sorted((out/'configs').glob('*.json'))})
    print(out)

if __name__=='__main__':main()
