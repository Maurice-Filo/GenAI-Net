"""Freeze existing evidence and render paper assets; never run scientific jobs."""

import ast
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

OLD = ROOT / 'paper/iclr2027_genai_net_llm'
REPORTS = ROOT / 'comparisons/search_exploration/reports'
ARMS = ['none', 'llm-to-rl', 'rl-to-llm', 'full']
LABELS = ['None', 'Insertion', 'Return', 'Full']
COLORS = ['#737373', '#cc7833', '#347fa3', '#93528b']
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source, name):
    target = HERE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    SOURCES[name] = {'source': str(source), 'sha256': digest(source)}
    return target


def json_asset(source, name):
    return json.loads(freeze(source, 'generated/' + name).read_text())


def save(fig, name):
    fig.savefig(HERE / 'figures' / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(HERE / 'figures' / (name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def table(name, header, rows, columns):
    text = '\\begin{tabular}{@{}' + columns + '@{}}\\toprule\n'
    text += ' & '.join(header) + r' \\ \midrule' + '\n'
    text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    text += '\n\\bottomrule\\end{tabular}\n'
    (HERE / 'generated' / name).write_text(text)


def canonical_count_figures(audit):
    """Split the audited figure by counting unit, without changing memberships."""
    bootstrap = np.random.default_rng(20260925).integers(0, 20, (20000, 20))
    jitter = np.random.default_rng(42).uniform(-.18, .18, 20)
    metrics = [('valid', 'Valid originals'), ('cores', 'Pruned cores'),
               ('robust_cores', 'Robust cores')]
    for pooled in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.65))
        for j, (key, title) in enumerate(metrics):
            ax = axes[j]
            for i, arm in enumerate(ARMS):
                values = np.array([r[key] for r in audit['rows'] if r['arm'] == arm])
                if len(values) != 20:
                    raise ValueError('Canonical plot requires 20 seeds per arm')
                center = (next(r[key] for r in audit['pooled'] if r['arm'] == arm)
                          if pooled else values.mean())
                ax.bar(i, center, color=COLORS[i], alpha=.8 if pooled else .2, width=.65)
                if not pooled:
                    low, high = np.quantile(values[bootstrap].mean(axis=1), [.025, .975])
                    ax.scatter(i+jitter, values, s=10, color=COLORS[i], alpha=.7)
                    ax.errorbar(i, center, yerr=[[center-low], [high-center]],
                                fmt='D', color='#222222', capsize=2, markersize=3)
                ax.text(i, 1.02, f'{center:.0f}' if pooled else f'{center:.1f}',
                        transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
            ax.set_title(f"{'ABC'[j]}  {title}", loc='left', pad=22)
            ax.set(xticks=range(4), xticklabels=LABELS, ylim=(0, None), xlim=(-.55, 3.55))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Unique across 20 seeds' if pooled else 'Unique per seed')
        fig.tight_layout(w_pad=1.2)
        save(fig, 'canonical_counts_pooled' if pooled else 'canonical_counts_per_seed')


def main():
    for directory in ('generated', 'figures', 'prompts'):
        (HERE / directory).mkdir(exist_ok=True)
    preservation = HERE / 'preservation_manifest.json'
    if not preservation.exists():
        preservation.write_text(json.dumps({str(p.relative_to(OLD)): digest(p)
            for p in sorted(OLD.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}, indent=2) + '\n')
    freeze(OLD / 'references.bib', 'references_inherited.bib')
    freeze(ROOT / 'Figure1.pdf', 'figures/full_duplex_architecture.pdf')
    rpa = json_asset(OLD / 'generated/rpa_optimizer_comparison.json', 'rpa_optimizers.json')
    logic = json_asset(REPORTS / 'logic5-four-arm-statistics-20260924/summary.json', 'logic_endpoints.json')
    robust = json_asset(REPORTS / 'logic5-portfolio-robustness-20260925/summary.json', 'logic_robustness.json')
    channels = json_asset(REPORTS / 'channel-effects-logic-rpa-20260925/summary.json', 'channels.json')
    json_asset(REPORTS / 'logic5-robust-provenance-20260925/summary.json', 'logic_provenance.json')
    rpacom = json_asset(Path('/local1/rossin/search-exploration/rpa-llm-interaction-20260918/topology-families-final20-r2/summary.json'), 'rpa_communication.json')
    canonical = json_asset(REPORTS / 'logic5-portfolio-robustness-20260925/canonical_audit.json', 'canonical_audit.json')
    freeze(OLD / 'figures/rpa_optimizer_robustness.pdf', 'figures/rpa_optimizer_robustness.pdf')
    freeze(REPORTS / 'logic5-robust-provenance-20260925/LOGIC5_ROBUST_PROVENANCE.pdf', 'figures/logic_provenance.pdf')
    freeze(REPORTS / 'channel-effects-logic-rpa-20260925/02_unfitted_rl_sensitivity.pdf', 'figures/unfitted_channels.pdf')
    for task, version in [('logic5', 'v2'), ('rpa', 'v1')]:
        freeze(ROOT / f'comparisons/search_exploration/prompts/{task}-evidence-exploitation-{version}.md', f'prompts/{task}.txt')
    protocol = ROOT / 'RL4CRN/llm/workspace_loss_protocol.py'
    constants = {}
    for node in ast.parse(protocol.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    for key in ('SYSTEM_PROMPT', 'DECIDER_TEMPLATE', 'WRITER_TEMPLATE'):
        name = f'prompts/{key.lower()}.txt'
        (HERE / name).write_text(constants[key])
        SOURCES[name] = {'source': str(protocol), 'constant': key,
                         'sha256': digest(HERE / name), 'module_sha256': digest(protocol)}
    sample = json.loads((Path(logic['endpoints'][0]['output']) / 'manifest.json').read_text())
    if hashlib.sha256(constants['SYSTEM_PROMPT'].encode()).hexdigest() != sample['system_prompt_sha256']:
        raise ValueError('Current system prompt differs from frozen Logic manifest')
    if digest(HERE / 'prompts/logic5.txt') != sample['task_prompt_sha256']:
        raise ValueError('Current Logic prompt differs from campaign')
    json_asset(Path(logic['endpoints'][0]['output']) / 'manifest.json', 'logic_example_manifest.json')
    # This is an existing final RPA run, not a new source of experimental evidence.
    rpamanifest = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1/runs/flash-adagrad-rpa-seed01/manifest.json')
    if not rpamanifest.exists():
        plan = json.loads(Path('/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1/plan.json').read_text())
        job = plan['jobs'][0]
        rpamanifest = Path(job.get('output', '/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1/runs/' + job['name'])) / 'manifest.json'
    rm = json_asset(rpamanifest, 'rpa_example_manifest.json')
    if rm['system_prompt_sha256'] != sample['system_prompt_sha256'] or digest(HERE/'prompts/rpa.txt') != rm['task_prompt_sha256']:
        raise ValueError('RPA prompt fingerprint mismatch')

    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.titlesize': 9})
    canonical_count_figures(canonical)

    optimizer_names = [('julia', 'AdaGrad hybrid'), ('evolution', 'Evolution-fit hybrid'), ('ga', 'Standalone GA')]
    table('optimizers.tex', ['Workflow', 'Median loss', 'Mean $N_q$', 'Qualifying runs', 'Attempts'], [
        [label, f"{np.median([r['loss'] for r in rpa['primary']['endpoints'] if r['arm']==arm]):.6f}",
         f"{np.mean([r['qualified_discovered'] for r in rpa['primary']['endpoints'] if r['arm']==arm]):.2f}",
         f"{sum(r['loss'] < .016491168958935352 for r in rpa['primary']['endpoints'] if r['arm']==arm)}/20",
         f"{min(r['evaluations'] for r in rpa['primary']['endpoints'] if r['arm']==arm):,}--{max(r['evaluations'] for r in rpa['primary']['endpoints'] if r['arm']==arm):,}"]
        for arm, label in optimizer_names], 'lrrrr')
    table('logic.tex', ['Arm', 'Median loss', 'Mean $N_q$', 'Mean cores', 'Mean robust cores'], [
        [label, f"{logic['summary'][arm]['median_loss']:.6f}", f"{logic['summary'][arm]['mean_qualified']:.2f}",
         f"{robust['groups'][arm]['cores']['mean']:.2f}", f"{robust['groups'][arm]['robust_cores']['mean']:.2f}"]
        for arm, label in zip(ARMS, LABELS)], 'lrrrr')
    short = dict(zip(ARMS, LABELS))
    table('logic_tests.tex', ['Comparison', 'Endpoint', 'W/T/L', 'Raw $p$', 'Holm$_{12}$ $p$'], [
        [short[t['a']] + ' vs ' + short[t['b']], 'Loss' if t['metric']=='loss' else '$N_q$',
         f"{t['wins']}/{t['ties']}/{t['losses']}", f"{t['p']:.6g}", f"{t['holm_p_12']:.6g}"]
        for t in logic['tests']], 'llrrr')
    table('rpa_tests.tex', ['AdaGrad versus', 'Endpoint', 'W/T/L', 'Holm$_4$ $p$'], [
        ['Evolution-fit' if t['control']=='evolution' else 'GA',
         'Loss' if t['metric']=='loss' else '$N_q$',
         f"{t['wins']}/{t['ties']}/{t['losses']}", f"{t['p_holm']:.6g}"]
        for t in rpa['primary']['contrasts']], 'llrr')
    perturbation = rpa['posthoc_perturbation_contrasts']
    p_rows = []
    for radius in sorted({t['radius'] for t in perturbation}):
        values = {(t['arm'],t['control']):t for t in perturbation if t['radius']==radius}
        a, b = values['julia','ga'], values['julia','evolution']
        p_rows.append([f"{100*(radius-1):.0f}\\%", f"{20*a['mean_a']:.2f}",
            f"{20*b['mean_b']:.2f}", f"{20*a['mean_b']:.2f}",
            f"{a['p_holm_15']:.5g}", f"{b['p_holm_15']:.5g}",
            f"{values['evolution','ga']['p_holm_15']:.5g}"])
    table('rpa_perturbation_tests.tex', ['Radius', 'A', 'E', 'G', 'A vs G', 'A vs E', 'E vs G'], p_rows, 'rrrrrrr')
    rpa_names = {'control': 'No-LLM hybrid', 'full': 'Full duplex', 'llm-to-rl': 'Insertion only',
                 'rl-to-llm': 'Return only', 'independent-union': 'Independent union'}
    table('rpa_communication.tex', ['Workflow', 'Median loss', 'Mean $N_q$', 'Mean families'], [
        [rpa_names[r['arm']], f"{r['median_best_loss']:.7f}", f"{r['mean_qualified_topologies']:.2f}", f"{r['mean_families']:.2f}"]
        for r in rpacom['arms']], 'lrrr')

    fig, axes = plt.subplots(2, 2, figsize=(6, 4.9))
    for i, task in enumerate(('logic', 'rpa')):
        for j, metric in enumerate(('llm_yield', 'rl_with_fit_loss')):
            ax = axes[i, j]
            rows = [r for r in channels['contrasts'] if r['task']==task and r['metric']==metric]
            ax.axvline(0, color='#999999', ls='--', lw=.8)
            for y, r in enumerate(rows):
                color = '#93528b' if task == 'logic' else '#168577'
                ax.plot(r['ci95'], [y, y], color=color, lw=2)
                ax.plot(r['effect'], y, 'D', color=color, ms=4)
                annotation = (f"Holm p={r['holm_p_30_logic']:.3f}" if task == 'logic' else 'Descriptive')
                ax.text(.98, y-.24, f"{r['effect']:+.2f}; {annotation}", ha='right', fontsize=6.5, transform=ax.get_yaxis_transform())
            ax.set(yticks=[0,1], yticklabels=['Return vs none', 'Full vs insertion'] if j==0 else ['Insertion vs none', 'Full vs return'], ylim=(1.5,-.65))
            ax.tick_params(axis='y', labelsize=7)
            ax.set_title(f"{task.upper()}: " + ('Search to LLM' if j==0 else 'LLM to search'), loc='left')
            ax.set_xlabel(('Additional robust cores' if task=='logic' else 'Additional direct-LLM\nqualifying topologies') if j==0 else 'RL-emitted loss reduction (%)')
            ax.margins(x=.25)
            ax.grid(axis='x', alpha=.15)
    fig.tight_layout(w_pad=1.2, h_pad=2)
    save(fig, 'channels')

    (HERE / 'generated/source_manifest.json').write_text(json.dumps(SOURCES, indent=2)+'\n')
    print(f'Frozen {len(SOURCES)} inputs; current communication cohorts only; no scientific calls.')


if __name__ == '__main__':
    main()
