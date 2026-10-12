"""Fast checks on frozen manuscript evidence, without simulations or model calls."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/jmlr_hybrid_crn'


def read(name):
    return json.loads((PAPER / 'generated' / name).read_text())


def test_frozen_asset_hashes():
    for name, row in read('source_manifest.json').items():
        assert hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == row['sha256'], name


def test_no_incompatible_historical_llm_comparison_in_manuscript():
    sources = '\n'.join(p.read_text() for p in (PAPER / 'sections').glob('*.tex'))
    for obsolete in ('fixed_proposal_motivation', 'generated/fixed_proposals',
                     'local-model banks', 'rpa_optimizer_balance'):
        assert obsolete not in sources
    assert not any('fixed_proposal' in name for name in read('source_manifest.json'))


def test_logic_tables_preserve_endpoint_definitions():
    endpoints = read('logic_endpoints.json')
    robust = read('logic_robustness.json')
    table = (PAPER / 'generated/logic.tex').read_text()
    for arm, summary in endpoints['summary'].items():
        assert summary['n'] == 20
        assert f"{summary['median_loss']:.6f}" in table
        assert f"{summary['mean_qualified']:.2f}" in table
        values = robust['groups'][arm]['robust_cores']['values']
        assert len(values) == 20
        assert f"{np.mean(values):.2f}" in table
    assert len(endpoints['tests']) == 12


def test_new_objective_and_prompt_fingerprints():
    method = (PAPER / 'sections/method.tex').read_text()
    assert r'\lambda_E=0.1' in method
    for name, task in [('logic_example_manifest.json', 'logic5'), ('rpa_example_manifest.json', 'rpa')]:
        manifest = read(name)
        assert manifest['withhold_initial_hof'] is True
        assert manifest['config']['imitation_weight'] == .1
        assert manifest['parameter_operator'] == 'julia_adagrad'
        assert hashlib.sha256((PAPER / 'prompts/system_prompt.txt').read_bytes()).hexdigest() == manifest['system_prompt_sha256']
        assert hashlib.sha256((PAPER / f'prompts/{task}.txt').read_bytes()).hexdigest() == manifest['task_prompt_sha256']


def test_no_cross_cohort_inference_promotion():
    channels = read('channels.json')
    assert all(t['holm_p_30_logic'] is None for t in channels['contrasts'] if t['task']=='rpa')
    rpa = read('rpa_optimizers.json')
    ga_loss = next(t for t in rpa['primary']['contrasts'] if t['control']=='ga' and t['metric']=='loss')
    assert ga_loss['p_holm'] > .05
    assert rpa['primary']['inference_ready']


def test_manuscript_has_no_external_latex_inputs():
    for path in [PAPER / 'main.tex', *sorted((PAPER / 'sections').glob('*.tex'))]:
        source = path.read_text()
        assert '/local1/' not in source
        assert '../iclr2027' not in source


def test_full_duplex_focus_and_original_opening_figure():
    main = (PAPER / 'main.tex').read_text()
    intro = (PAPER / 'sections/introduction.tex').read_text()
    assert r'\title{Full-Duplex Communication Between Language Models and Hybrid Search for the Design of Biomolecular Reaction Networks}' in main
    assert intro.index('full_duplex_architecture.pdf') < intro.index(r'\subsection{Related Work}')
    assert r'\label{fig:architecture}' in intro
    assert (PAPER / 'figures/full_duplex_architecture.pdf').read_bytes() == (ROOT / 'Figure1.pdf').read_bytes()
    for section in ('method', 'discussion'):
        assert 'Full-duplex' in (PAPER / f'sections/{section}.tex').read_text() or 'Full-Duplex' in (PAPER / f'sections/{section}.tex').read_text()


def test_canonical_count_rows_are_split_without_changing_endpoints():
    main_results = (PAPER / 'sections/communication_results.tex').read_text()
    appendix = (PAPER / 'sections/appendix.tex').read_text()
    assert 'figures/canonical_counts_per_seed.pdf' in main_results
    assert 'figures/canonical_counts_pooled.pdf' not in main_results
    assert 'figures/canonical_counts_pooled.pdf' in appendix
    assert 'figures/canonical_counts_per_seed.pdf' not in appendix
    audit = read('canonical_audit.json')
    robust = read('logic_robustness.json')
    for arm, group in robust['groups'].items():
        rows = sorted((r for r in audit['rows'] if r['arm'] == arm), key=lambda r: r['seed'])
        assert len(rows) == 20
        for metric in ('cores', 'robust_cores'):
            assert [r[metric] for r in rows] == group[metric]['values']
    assert {r['arm']: r['robust_cores'] for r in audit['pooled']} == {
        'none': 107, 'llm-to-rl': 106, 'rl-to-llm': 152, 'full': 139}
