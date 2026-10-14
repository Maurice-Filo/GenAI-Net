"""Stage compact diagnostic records and exact campaign instructions, without secrets."""
import argparse
import json
import os
from pathlib import Path
from build_release import copy_file, dump, SMALL, REPO, PAPER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output.resolve()
    assert (root/'RELEASE_BUILD').exists()
    sources = {}
    def copy(src, rel):
        if not src.is_file():
            return
        copy_file(src, root/rel)
        sources[str(src)] = str(rel)
    jobs = json.loads((root/'manifest/runs.json').read_text())
    # Stage small endpoint files now, independently of database compression.
    campaigns = set()
    for j in jobs:
        src = Path(j['source'])
        for name in SMALL:
            dst = Path('data/runs')/j['id']/name
            if not (root/dst).exists():
                copy(src/name, dst)
        for parent in [src, *src.parents]:
            if (parent/'plan.json').exists():
                campaigns.add(parent)
                break
    for campaign in sorted(campaigns):
        rel = Path('data/campaigns')/campaign.name
        for name in ['plan.json', 'active_plan.json', 'TASK.md', 'reference_TASK.md',
                     'prompt.diff', 'transport.diff', 'configuration_matches.json', 'input_hashes.json']:
            copy(campaign/name, rel/name)
        config = campaign/'configs'
        if config.is_dir():
            for p in config.rglob('*'):
                if p.is_file() and p.suffix in ['.json', '.toml', '.yaml', '.txt', '.md']:
                    copy(p, rel/'configs'/p.relative_to(config))
        source = campaign/'source'
        if source.is_dir():
            for p in source.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and (
                    p.suffix in ['.py', '.jl', '.json', '.md', '.txt', '.toml', '.yaml', '.yml', '.sh', '.mjs', '.js']
                    or p.name == 'scientific_source.tar.gz'):
                    copy(p, rel/'source'/p.relative_to(source))
    evidence = [
        ('logic', Path('/local1/rossin/search-exploration/logic5-portfolio-diagnostics-order-fallback-20260924')),
        ('rpa', Path('/local1/rossin/search-exploration/rpa-llm-interaction-20260918/topology-families-final20-r2')),
    ]
    for name, src in evidence:
        for p in src.rglob('*'):
            if not p.is_file() or p.suffix not in ['.json', '.gz', '.csv', '.txt']:
                continue
            if any(x in p.relative_to(src).parts for x in ['logs', 'controller', '__pycache__']):
                continue
            copy(p, Path('data/diagnostics')/name/p.relative_to(src))
    p = Path('/local1/rossin/search-exploration/rpa-llm-interaction-20260918/provenance/provenance.json')
    copy(p, Path('data/diagnostics/rpa/provenance.json'))
    p = REPO/'comparisons/search_exploration/reports/rpa-communication-portfolio-20260925/diagnostics/selection.json.gz'
    copy(p, Path('data/diagnostics/rpa/selection.json.gz'))
    for p in (PAPER/'prompts').rglob('*'):
        if p.is_file() and p.suffix in ['.md', '.txt', '.json']:
            copy(p, Path('data/prompts')/p.relative_to(PAPER/'prompts'))
    for directory in ['RL4CRN', 'comparisons/llm_crn_generation', 'tools', 'comparisons/search_exploration/prompts']:
        for p in (REPO/directory).rglob('*'):
            if p.is_file() and p.suffix == '.md' and not any(x in p.parts for x in ['node_modules', '.git']):
                copy(p, p.relative_to(REPO))
    # The production AdaGrad service used this external Julia project.
    julia = Path('/local0/rossin/git/SynthEvo/SynthEvo')
    for p in julia.rglob('*'):
        if p.is_file() and p.suffix in ['.jl', '.toml']:
            copy(p, Path('numerical_runtime/SynthEvo')/p.relative_to(julia))
    for n in ['README.md', 'OPTIMIZATION.md']:
        copy(julia.parent/n, Path('numerical_runtime')/n)
    for n in ['manifest.json', 'scientific_source.tar.gz']:
        rel = Path('comparisons/search_exploration/baseline')/n
        copy(REPO/rel, rel)
    dump(root/'manifest/evidence_paths.json', sources)
    print(f'Staged {len(sources)} evidence files from {len(campaigns)} campaigns', flush=True)


if __name__ == '__main__':
    main()
