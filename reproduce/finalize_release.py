"""Write the human-readable cohort index and checksums after validation."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from build_release import sha, dump, REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output.resolve()
    assert (root/'RELEASE_BUILD').exists()
    p = root/'analysis/biomolecular_full_duplex_rewrite'
    mapping = json.loads((root/'manifest/path_map.json').read_text())
    runs = json.loads((root/'manifest/runs.json').read_text())
    cohorts = []
    for run in runs:
        for membership in run['memberships']:
            cohorts.append(dict(selection='LLM resource accounting', group=membership['group'],
                                seed=membership['seed'], run=run['id']))
    for selection, path in [('GenAI-Net comparison', p/'genainet_comparison/audit.json'),
                            ('Numerical optimization', p/'numerical_comparison/frozen_runs.json')]:
        for row in json.loads(path.read_text())['runs']:
            cohorts.append(dict(selection=selection, group=row.get('task', 'rpa')+'/'+row['arm'],
                                seed=row['seed'], run=mapping[row['folder']]))
    for row in json.loads((root/'analysis/jmlr_hybrid_crn/generated/channels.json').read_text())['rows']:
        cohorts.append(dict(selection='Communication ablation', group=row['task']+'/'+row['arm'],
                            seed=row['seed'], run=mapping[row['folder']]))
    with (root/'manifest/cohorts.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['selection', 'group', 'seed', 'run'])
        w.writeheader(); w.writerows(cohorts)
    groups = Counter((r['selection'], r['group']) for r in cohorts)
    lines = ['Cohort selections (overlapping selections must not be summed)',
             '==========================================================', '']
    lines.extend(f'{selection} | {group}: {n} runs' for (selection, group), n in sorted(groups.items()))
    lines += ['', 'manifest/cohorts.csv maps every listed seed to its compact run directory.',
              'Figure inputs are frozen in analysis/ and retain their original source hashes.',
              'Read manifest/figures.json for the exact manuscript figure list.',
              'Recorded source paths are provenance strings, not paths needed by reproduce.py.']
    (root/'COHORTS.txt').write_text('\n'.join(lines)+'\n')
    figure_map = {}
    for name in json.loads((root/'manifest/figures.json').read_text()):
        if name == 'figures/full_duplex_architecture.pdf':
            script, inputs = None, ['expected/figures/'+name]
        elif name.startswith('eil_ablation/'):
            script, inputs = 'plot_eil_ablation.py', ['eil_ablation/endpoints.csv', 'eil_ablation/curves.csv']
        elif name.startswith('token_costs/'):
            script, inputs = 'token_costs/plot_resources.py', ['token_costs/summary.json', 'token_costs/tool_calls_summary.json']
        elif name == 'convergence/search_archive_convergence.pdf':
            script, inputs = 'plot_convergence.py', ['convergence/snapshots.csv']
        elif name == 'figures/logic_provenance.pdf':
            script, inputs = 'plot_logic_provenance.py', ['../jmlr_hybrid_crn/generated/logic_provenance.json']
        elif name.startswith('parity_performance/'):
            script = 'parity_performance/'+('plot_reactions.py' if 'reactions' in name else 'plot_qualification.py')
            inputs = ['parity_performance/qualification_snapshot.json', 'parity_performance/convergence_snapshot.json',
                      'parity_performance/reaction_network_snapshot.json']
        elif 'qwen_rpa_timeout' in name:
            script, inputs = 'qwen_comparison/plot_timeout_supplement.py', ['qwen_comparison/rpa_timeout_portfolios.json',
                'qwen_comparison/qwen_rpa_timeout3600_snapshot.json', 'qwen_comparison/qwen_preliminary_convergence_snapshot.json',
                'genainet_comparison/audit.json']
        elif 'prompt_ablation' in name:
            script, inputs = 'plot_prompt_ablation.py', ['prompt_ablation/final_20seed/snapshot.json']
        else:
            script = 'proposed_results_figures.py'
            inputs = ['numerical_comparison/frozen_runs.json', 'numerical_comparison/aggregate_curves.csv',
                'convergence/snapshots.csv', 'genainet_comparison/audit.json',
                'qwen_comparison/qwen_preliminary_convergence_snapshot.json',
                'results_figure_proposal/rpa_integral_screen.csv', 'results_figure_proposal/rpa_provenance_snapshot.json',
                'results_figure_proposal/response_snapshots.json', '../jmlr_hybrid_crn/generated/canonical_audit.json',
                '../jmlr_hybrid_crn/generated/logic_provenance.json']
        assert all((p/n).exists() for n in inputs) if script else True
        figure_map[name] = dict(script=script, input_paths_relative_to_analysis_paper_directory=inputs,
                               command='python reproduce.py figures')
    dump(root/'manifest/figure_inputs.json', figure_map)

    # Retain packaging code so the selection/export procedure is auditable too.
    for source in (REPO/'reproduce').glob('*.py'):
        dest = root/'release_tools'/source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
    # Snapshot the successful checks as evidence, separately from user outputs.
    for name in ['verification.json', 'scientific_checks.json', 'figure_report.json', 'eil_statistics_recomputed.json']:
        src = root/'output'/name
        assert src.exists(), name
        shutil.copyfile(src, root/'manifest'/name)
    figures = json.loads((root/'manifest/figure_report.json').read_text())
    assert len(figures) == 17 and all(r.get('pixel_equal') for r in figures)
    assert json.loads((root/'manifest/verification.json').read_text())['all_passed']
    exported = [json.loads((root/'data/runs'/r['id']/'export.json').read_text()) for r in runs]
    assert len(exported) == 521
    files = [f for f in root.rglob('*') if f.is_file() and not
             any(part in ['.venv', '__pycache__', 'output'] for part in f.relative_to(root).parts)
             and f != root/'manifest/checksums.json']
    # Detect obvious accidental credential artifacts without printing contents.
    suspect = []
    for f in files:
        rel = f.relative_to(root)
        if f.name in ['.env', '.credentials.yaml', 'credentials.json', 'id_rsa', 'id_ed25519']:
            suspect.append(str(rel))
        if f.suffix in ['.json', '.jsonl', '.yaml', '.yml', '.md', '.txt'] and f.stat().st_size < 10*1024**2:
            if re.search(rb'\bsk-[A-Za-z0-9_-]{24,}\b', f.read_bytes()):
                suspect.append(str(rel))
    assert not suspect, 'Review possible credentials in: '+', '.join(suspect)
    summary = dict(unique_runs=len(runs), database_runs=sum(bool(d['tables']) for d in exported),
        legacy_runs=sum(not d['tables'] for d in exported),
        source_database_bytes=sum(d.get('source_database_bytes', 0) for d in exported),
        exported_rows_by_table=dict(sum((Counter({k: v['rows'] for k, v in d['tables'].items()}) for d in exported), Counter())),
        payload_bytes=sum(f.stat().st_size for f in files), payload_files=len(files),
        figures=17, figures_pixel_identical=True,
        omitted=['full trajectory/cache blobs', 'policy checkpoints', 'temporary numerical workspaces',
                 'full LLM transcripts', 'manuscript'],
        obvious_credential_scan='no findings; personal/source paths remain',
        new_search_runtime='Sources and settings retained; offline analysis validated, new full campaigns not rerun')
    dump(root/'manifest/payload_summary.json', summary)
    files = [f for f in files if f != root/'manifest/payload_summary.json']+[root/'manifest/payload_summary.json']
    checksums = {str(f.relative_to(root)): sha(f) for f in sorted(files)}
    dump(root/'manifest/checksums.json', checksums)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
