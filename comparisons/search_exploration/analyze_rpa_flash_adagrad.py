"""Saved-results-only comparison of full-duplex Flash and the AdaGrad/RL control."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

import numpy as np

from comparisons.search_exploration.analyze_rpa_statistical import holm, paired

FULL = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1')
CONTROL = Path('/local1/rossin/search-exploration/rpa-statistical-20seed-20260915')
ARMS = ('control', 'flash')
LABELS = {'control': 'AdaGrad/RL', 'flash': '+ Flash (full duplex)'}
COLORS = {'control': '#168577', 'flash': '#a64e85'}


def read(path):
    return json.loads(path.read_text())


def builder_hash(folder):
    with tarfile.open(folder / 'source.tar.gz') as archive:
        name = next(n for n in archive.getnames() if n.endswith('RL4CRN/search_exploration/evaluation.py'))
        tree = ast.parse(archive.extractfile(name).read().decode())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def topology_losses(records):
    """Count structures once, retaining their best complete canonical score."""
    best = {}
    for r in records:
        if not r['simulation_complete']:
            continue
        loss = float(r['loss'])
        if not np.isfinite(loss) or loss < 0:
            raise ValueError('Invalid completed canonical loss')
        key = r['topology_hash']
        best[key] = min(best.get(key, float('inf')), loss)
    return sorted(best.values())


def extract(folder, arm, seed):
    m, c = read(folder / 'manifest.json'), read(folder / 'completed.json')
    cfg, metrics, counters = m['config'], c['metrics'], c['counters']
    if c['status'] != 'complete' or c['epochs'] != 100 or cfg['seed'] != seed or m['task'] != 'rpa':
        raise ValueError(f'Incomplete or mismatched run: {folder}')
    if (counters['expensive_evaluations'] > cfg['evaluation_budget'] or
            counters.get('julia/reserved', 0) or metrics.get('Optimization/pending', 0) or
            metrics.get('LLM/pending_rounds', 0) or c.get('pending_optimization_jobs', 0)):
        raise ValueError(f'Budget violation or undrained work: {folder}')
    if arm == 'flash' and (m['communication_mode'] != 'full' or not m['withhold_initial_hof']):
        raise ValueError('Incorrect full-duplex protocol')
    if arm == 'control' and m['harness']:
        raise ValueError('Control contains an LLM')
    path = (folder / 'exploration.sqlite').resolve()
    wal = path.with_name(path.name + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Refusing immutable reads of an uncheckpointed database')
    threshold = cfg['quality_threshold']
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        db.execute('PRAGMA temp_store=MEMORY')
        rows = db.execute("""SELECT json_extract(payload,'$.topology_hash'),
            MIN(json_extract(payload,'$.loss')) FROM designs
            WHERE json_extract(payload,'$.simulation_complete')=1
            GROUP BY json_extract(payload,'$.topology_hash')""").fetchall()
        losses = topology_losses({'topology_hash': key, 'loss': loss,
                                  'simulation_complete': True} for key, loss in rows)
        for (payload,) in db.execute("""SELECT payload FROM designs
                WHERE json_extract(payload,'$.simulation_complete')=1
                AND json_extract(payload,'$.loss') < ?""", (threshold,)):
            record = json.loads(payload)
            ids, rates = record['reaction_ids'], record['rates']
            if (len(ids) != 5 or len(set(ids)) != 5 or set(ids) & {0, 1, 28} or
                    len(rates) != 5 or not all(.001 <= k <= 100 for k in rates) or
                    not record['qualified']):
                raise ValueError('Qualifying design violates constraints or qualification flag')
    if not losses or not np.isclose(losses[0], metrics['Archive/best_loss'], rtol=1e-12, atol=0):
        raise ValueError('Historical discoveries and endpoint best loss disagree')
    records = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    by_epoch = {row['epoch']: row for row in records}
    if set(by_epoch) != set(range(101)):
        raise ValueError('Missing training or terminal metric boundary')
    history = read(folder / 'llm_history.json') if arm == 'flash' else []
    endpoint = dict(arm=arm, seed=seed, folder=str(folder), loss=losses[0],
        qualified_discovered=int(np.searchsorted(losses, threshold, side='left')),
        qualified_retained=metrics['Archive/qualified_topologies'],
        evaluations=counters['expensive_evaluations'],
        canonical_evaluations=counters['expensive_evaluations'] - counters.get('julia/joint_evaluations', 0),
        joint_evaluations=counters.get('julia/joint_evaluations', 0),
        minutes=c['elapsed_seconds'] / 60, failed_fit_jobs=metrics.get('Fit/failed_jobs_cumulative', 0),
        llm_rounds=len(history), llm_failed_rounds=sum(bool(h.get('error')) for h in history),
        llm_valid_members=sum(h.get('valid', 0) for h in history),
        llm_late_rounds=sum(h.get('completed_epoch', 0) >= 100 for h in history),
        evaluation_seconds=cfg['evaluation_seconds'], fit_seconds=cfg['fit_seconds'],
        search_workers=cfg['workers'], optimization_workers=m.get('optimization_workers'),
        loss_disagreement_policy=m['julia_recipe'].get('loss_disagreement_policy', 'abort'),
        manifest_sha256=hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
        completed_sha256=hashlib.sha256((folder / 'completed.json').read_bytes()).hexdigest())
    return {'endpoint': endpoint, 'losses': losses, 'trace': [by_epoch[i] for i in range(101)],
            'manifest': m, 'builder_hash': builder_hash(folder)}


def matched_settings(a, b):
    differences = {k: [a['config'].get(k), b['config'].get(k)] for k in
                   a['config'].keys() | b['config'].keys() if a['config'].get(k) != b['config'].get(k)}
    if set(differences) - {'workers', 'evaluation_seconds', 'fit_seconds'}:
        raise ValueError(f'Unexpected scientific configuration difference: {differences}')
    for key in ('task_config', 'post_prune', 'parameter_operator'):
        if a[key] != b[key]:
            raise ValueError(f'Mismatch in {key}')
    if a['policy']['deep_layer_size'] != b['policy']['deep_layer_size']:
        raise ValueError('Different policy embedding size')
    for key in a['policy'].keys() & b['policy'].keys():
        if a['policy'][key] != b['policy'][key]:
            raise ValueError(f'Different policy {key}')
    # Both runners use the same builder; the newer manifest omits default width/depth.
    for source in ('RL4CRN/search_exploration/session.py',):
        if a.get('new_source', {}).get(source) != b.get('new_source', {}).get(source):
            raise ValueError(f'Different policy/task construction source: {source}')
    recipe_a, recipe_b = dict(a['julia_recipe']), dict(b['julia_recipe'])
    recipe_a.pop('loss_disagreement_policy', None)
    recipe_b.pop('loss_disagreement_policy', None)
    if recipe_a != recipe_b:
        raise ValueError('Different fitting recipe')
    return differences


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output, runs, tests, threshold):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    groups = {arm: [r for r in runs if r['endpoint']['arm'] == arm] for arm in ARMS}
    values = lambda arm, key: np.array([r['endpoint'][key] for r in groups[arm]])
    pdf = PdfPages(output / 'RPA_FLASH_VS_ADAGRAD.pdf')

    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=15, x=.06, ha='left', y=.99)
        fig.text(.06, .015, note, fontsize=9, va='bottom')
        fig.tight_layout(rect=(.025, .12, .995, .92), w_pad=2.3)
        fig.savefig(output / (name + '.png'), dpi=200)
        fig.savefig(output / (name + '.pdf'))
        pdf.savefig(fig)
        plt.close(fig)

    def paired_dots(ax, key, ylabel, log=False):
        jitter = np.random.default_rng(42).uniform(-.075, .075, 20)
        a, b = [values(arm, key) for arm in ARMS]
        for j, av, bv in zip(jitter, a, b):
            ax.plot([j, 1+j], [av, bv], color='.8', lw=.8, zorder=1)
        for i, arm in enumerate(ARMS):
            v = values(arm, key)
            ax.scatter(i+jitter, v, color=COLORS[arm], s=25, zorder=2)
            ax.hlines(np.median(v), i-.19, i+.19, color=COLORS[arm], lw=3)
        ax.set_xticks([0, 1], ['AdaGrad/RL', '+ Flash\nfull duplex'])
        ax.set(xlim=(-.4, 1.4), ylabel=ylabel)
        if log:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.15)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.1))
    paired_dots(axes[0], 'loss', 'Best canonical loss (log scale)', True)
    paired_dots(axes[1], 'qualified_discovered', 'Distinct qualifying topologies discovered')
    for ax, key, heading in zip(axes, ('loss', 'qualified_discovered'), ('A  Better solutions', 'B  Larger useful sets')):
        s = tests[key]
        ax.set_title(f"{heading}\nW/T/L {s['wins']}/{s['ties']}/{s['losses']}  |  Holm p={s['p_holm']:.3g}", loc='left')
    d = values('flash', 'qualified_discovered') - values('control', 'qualified_discovered')
    order = np.argsort(d)
    axes[2].bar(np.arange(20), d[order], color=[COLORS['flash'] if x >= 0 else '#626970' for x in d[order]])
    axes[2].axhline(0, color='.3', lw=.8)
    axes[2].set(xticks=np.arange(20), xticklabels=order+1, xlabel='Matched seed (sorted by gain)',
                ylabel='Additional qualifying topologies', title='C  Consistency across seeds')
    axes[2].tick_params(axis='x', labelsize=8)
    save(fig, '01_paired_endpoints', 'RPA | Full-duplex Flash added to the AdaGrad/RL search',
         f'20 matched seeds; 100 epochs. Qualification: loss < {threshold:.5g}. Lines connect seeds; thick marks show medians.\n'
         'Counts deduplicate topology hashes over all saved discoveries, not only retained elites. Holm correction covers the two endpoint tests.')

    grid = np.unique(np.r_[np.geomspace(.0002, .08, 200), threshold])
    curves = {arm: np.array([np.searchsorted(r['losses'], grid, side='left') for r in groups[arm]]) for arm in ARMS}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
    curve_rows = []
    for arm in ARMS:
        mean, sd = curves[arm].mean(0), curves[arm].std(0, ddof=1)
        axes[0].plot(grid, mean, color=COLORS[arm], label=LABELS[arm], lw=2)
        axes[0].fill_between(grid, np.maximum(0, mean-sd), mean+sd, color=COLORS[arm], alpha=.13)
        curve_rows.extend(dict(arm=arm, threshold=float(t), mean=float(m), sd=float(s)) for t,m,s in zip(grid,mean,sd))
    delta = curves['flash'] - curves['control']
    ix = np.random.default_rng(20260917).integers(0, 20, (10000, 20))
    boot = delta[ix].mean(1)
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    axes[1].plot(grid, delta.mean(0), color=COLORS['flash'], lw=2)
    axes[1].fill_between(grid, lo, hi, color=COLORS['flash'], alpha=.18)
    axes[1].axhline(0, color='.35', lw=.8)
    for ax in axes:
        ax.set_xscale('log')
        ax.axvline(threshold, color='.3', ls='--', lw=1)
        ax.set_xlabel('Qualifying loss threshold (log scale)')
        ax.grid(alpha=.12)
    axes[0].set(ylabel='Mean distinct qualifying topologies', title='A  Quality-versus-yield trade-off', ylim=(0, None))
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(ylabel='Mean additional topologies with Flash', title='B  Paired gain at each threshold')
    save(fig, '02_threshold_sensitivity', 'RPA | Is the portfolio gain threshold-dependent?',
         'Dashed line: frozen qualification threshold. Left shading: +/-1 sample SD across seeds (not significance).\n'
         'Right shading: pointwise 95% paired bootstrap intervals, not simultaneous bands. This threshold sweep is exploratory.')
    write_csv(output / 'threshold_curves.csv', curve_rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.1))
    x = np.r_[np.arange(1, 101), 106]
    for arm in ARMS:
        for ax, key, median in ((axes[0], 'Archive/best_loss', True), (axes[1], 'Archive/qualified_topologies', False)):
            data = np.array([[r[key] for r in run['trace']] for run in groups[arm]])
            center = np.median(data, 0) if median else data.mean(0)
            lo, hi = np.quantile(data, [.25, .75], axis=0) if median else (np.maximum(0, center-data.std(0, ddof=1)), center+data.std(0, ddof=1))
            ax.plot(x[:100], center[:100], color=COLORS[arm], label=LABELS[arm], lw=2)
            ax.fill_between(x[:100], lo[:100], hi[:100], color=COLORS[arm], alpha=.13)
            ax.plot(x[99:], center[99:], color=COLORS[arm], ls=':', lw=1)
            ax.errorbar(106, center[-1], yerr=[[center[-1]-lo[-1]], [hi[-1]-center[-1]]],
                        color=COLORS[arm], fmt='o', ms=4, capsize=3)
    for ax in axes[:2]:
        ax.set(xlabel='Completed training epochs; T = drained endpoint', xticks=[0,25,50,75,100,106],
               xticklabels=['0','25','50','75','100','T'])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(alpha=.15)
    axes[0].set(yscale='log', ylabel='Median best canonical loss', title='A  Search progress')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(ylabel='Mean qualifying topologies retained', title='B  Useful archive during training', ylim=(0,None))
    paired_dots(axes[2], 'evaluations', 'Canonical + joint loss/gradient calls')
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda n, _: f'{n/1000:g}k'))
    axes[2].set_title('C  Actual numerical expenditure', loc='left')
    save(fig, '03_progress_and_compute', 'RPA | Training progress and numerical work',
         'A: median with interquartile band. B: mean +/-1 sample SD; retained counts can decrease and differ from all discoveries.\n'
         'T includes late model/fitting results, not further RL updates. Same cap is not equal consumed compute; LLM inference is additional.')
    pdf.close()


def analyze(full, control, output):
    fp, cp = read(full / 'plan.json'), read(control / 'active_plan.json')
    reference = read(control / cp['analysis_dir'] / 'summary.json')
    if not reference['inference_ready']:
        raise ValueError('Historical controls have not passed their cohort audit')
    controls = {j['seed']: Path(j.get('output', control / 'runs' / j['name'])) for j in cp['jobs'] if j['arm'] == 'julia'}
    flashes = {j['seed']: full / 'runs' / j['name'] for j in fp['jobs']}
    if set(controls) != set(range(1,21)) or set(flashes) != set(controls):
        raise ValueError('Expected exactly twenty matched seeds')
    runs, differences = [], []
    for seed in range(1,21):
        a, b = extract(controls[seed], 'control', seed), extract(flashes[seed], 'flash', seed)
        if a['builder_hash'] != b['builder_hash']:
            raise ValueError('Task/policy builders differ in archived run sources')
        differences.append(dict(seed=seed, config=matched_settings(a['manifest'], b['manifest'])))
        audited = next(r for r in reference['endpoints'] if r['arm']=='julia' and r['seed']==seed)
        for key in ('loss', 'qualified_discovered', 'qualified_retained', 'evaluations'):
            if not np.isclose(a['endpoint'][key], audited[key], rtol=1e-12, atol=0):
                raise ValueError(f'Historical extraction mismatch: seed {seed}, {key}')
        runs.extend([a,b])
        print(f"Seed {seed:02}: discoveries {a['endpoint']['qualified_discovered']} -> {b['endpoint']['qualified_discovered']}", flush=True)
    endpoint = [r['endpoint'] for r in runs]
    tests = {}
    for key in ('qualified_discovered', 'loss'):
        tests[key] = paired([r[key] for r in endpoint if r['arm']=='flash'],
                            [r[key] for r in endpoint if r['arm']=='control'], smaller=key=='loss')
    for key, p in zip(tests, holm([v['p'] for v in tests.values()])):
        tests[key]['p_holm'] = p
    threshold = runs[0]['manifest']['config']['quality_threshold']
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(post_hoc=True, new_simulations=0, new_model_calls=0, threshold=threshold,
                   contrasts=tests, endpoints=endpoint, configuration_differences=differences)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    write_csv(output / 'endpoints.csv', endpoint)
    plot(output, runs, tests, threshold)
    lines = ['# RPA: Full-Duplex Flash versus AdaGrad/RL', '',
        'All 20 matched seeds completed, 100 epochs each. Saved artifacts only; no simulations or model calls.', '',
        '| Workflow | Median best loss | Mean qualifying discoveries | Median qualifying discoveries | Mean retained qualifying | Mean numerical calls |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rr = [r for r in endpoint if r['arm']==arm]
        fields = [np.median([r['loss'] for r in rr]), np.mean([r['qualified_discovered'] for r in rr]),
                  np.median([r['qualified_discovered'] for r in rr]), np.mean([r['qualified_retained'] for r in rr]),
                  np.mean([r['evaluations'] for r in rr])]
        lines.append('| '+LABELS[arm]+' | '+' | '.join(f'{v:.6g}' for v in fields)+' |')
    lines += ['', '## Paired Evidence', '',
        'Effects are Flash minus control; ratios are Flash/control. Tests are two-sided Wilcoxon signed-rank with exact sign permutations and Pratt zeros; Holm correction covers these two endpoint tests. Bootstrap resamples matched seeds (10,000 resamples). These are post-hoc analyses, not corrections across the entire research program.', '']
    for key, s in tests.items():
        lines.append(f"- **{key}:** W/T/L {s['wins']}/{s['ties']}/{s['losses']}; mean difference {s['mean_difference']:.6g}, 95% paired bootstrap CI {s['mean_difference_ci95']}; median paired ratio {s['median_paired_ratio']:.6g}; raw p={s['p']:.6g}, Holm p={s['p_holm']:.6g}.")
    lines += ['', '## Interpretation and Limits', '',
        f'Qualification is complete canonical loss < {threshold:.17g}. One best score per complete structural topology hash is counted over all saved discoveries; rates and reaction permutations do not create new topologies. This is useful-set size, not evidence of mechanistic diversity or robustness.', '',
        'The comparison assesses adding the complete Flash/Harness pathway to the AdaGrad/RL search recipe (which already includes archive mutation and random proposals). It cannot isolate inward insertion, outward conditioning, imitation learning or parameter fitting. The directional arms and independent-union analysis address narrower questions separately.', '',
        'Both cohorts share task, bounds, batch size, NN, epochs, fitting recipe and numerical cap. Actual consumed evaluations differ. Historical worker allocations and concurrent load differ; some early historical controls retain finite timeout settings and the older loss-disagreement policy. Their prior completed-cohort audit is retained. Scheduling can change fitting admission and model insertion opportunities. Therefore this is not a pure communication-only or equal-wall-clock ablation. LLM inference is additional and is not represented by numerical-call counts.', '',
        'The historical active plan selects the final audited attempts, including infrastructure-recovery attempts; failed attempts are not pooled and their extra work is outside the endpoint compute panel. Candidate/fitting failures within successful runs are not discarded (see endpoints.csv). Final draining can improve terminal results without affecting completed RL updates; progress plots identify that boundary.', '',
        f"Flash: {sum(r['llm_rounds'] for r in endpoint)} completed/failed request records, {sum(r['llm_failed_rounds'] for r in endpoint)} failed request records, {sum(r['llm_valid_members'] for r in endpoint)} valid members (not unique structures), {sum(r['llm_late_rounds'] for r in endpoint)} request records collected at/after epoch 100.", '',
        '## Figures', '', '![Paired endpoints](01_paired_endpoints.png)', '',
        '![Threshold sensitivity](02_threshold_sensitivity.png)', '',
        '![Progress and numerical work](03_progress_and_compute.png)', '',
        'Combined vector PDF: [RPA_FLASH_VS_ADAGRAD.pdf](RPA_FLASH_VS_ADAGRAD.pdf).', '',
        'Reproduce from the repository:', '', '```bash',
        f'.venv/bin/python -m comparisons.search_exploration.analyze_rpa_flash_adagrad --full {full} --control {control} --output {output}', '```', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(tests, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', type=Path, default=FULL)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--output', type=Path, default=FULL / 'analysis-vs-adagrad-20260917')
    args = parser.parse_args()
    analyze(args.full, args.control, args.output)
