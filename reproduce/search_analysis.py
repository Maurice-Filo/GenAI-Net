"""Read-only summaries of fresh campaigns, separate from frozen-paper figures."""
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def jsonlines(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def valid(record):
    value = record.get('loss')
    return record.get('simulation_complete', False) and isinstance(value, (int, float)) and math.isfinite(value)


def records(output):
    """Keep rate-specific observations; deduplicate topologies only in summaries."""
    first, routes = {}, defaultdict(set)
    dbpath = output / 'exploration.sqlite'
    if not dbpath.exists():
        return []
    with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
        for (payload,) in db.execute("SELECT payload FROM events WHERE kind='archive_insertion' ORDER BY id"):
            row = json.loads(payload)
            key = row.get('candidate_hash')
            if key:
                first.setdefault(key, int(row.get('epoch', 0)) + 1)
                routes[key].add((str(row.get('origin', 'unknown')), str(row.get('source', 'unknown')),
                                 str(row.get('parameter_operator') or 'none')))
        result = []
        for key, payload in db.execute('SELECT key,payload FROM designs'):
            row = json.loads(payload)
            if valid(row):
                row['_checkpoint'] = first.get(row.get('candidate_hash', key))
                row['_routes'] = routes.get(row.get('candidate_hash', key), set())
                row['_record_stream'] = 'numerical_archive'
                result.append(row)
    for proposal in jsonlines(output / 'direct_llm_proposals.jsonl'):
        row = proposal.get('record')
        if row and valid(row):
            row = dict(row)
            row['_checkpoint'] = int(proposal['completed_epoch']) + 1
            row['_record_stream'] = 'direct_llm'
            result.append(row)
    return result


def summarize(rows, threshold):
    topologies = {}
    for row in rows:
        if not valid(row) or row['loss'] >= threshold:
            continue
        key = tuple(sorted(row['reaction_ids']))
        item = topologies.setdefault(key, dict(reaction_ids=list(key), loss=row['loss'],
                        categories=set(), observed_routes=set(), record_streams=set()))
        item['loss'] = min(item['loss'], row['loss'])
        if row.get('region') is not None:
            item['categories'].add(str(row['region']))
        item['record_streams'].add(row.get('_record_stream', 'unknown'))
        item['observed_routes'].add((str(row.get('origin', 'unknown')),
                                    str(row.get('source', 'unknown')),
                                    str(row.get('parameter_operator') or 'none')))
        item['observed_routes'].update(row.get('_routes', []))
    categories = set().union(*(x['categories'] for x in topologies.values())) if topologies else set()
    return topologies, categories


def csv_write(path, rows):
    if rows:
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def analyze_campaign(root, rpa_screen=False):
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.matplotlib'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO))
    campaign = read(root / 'campaign.json')
    dest = root / 'analysis'; dest.mkdir(exist_ok=True)
    endpoints, curves, memberships, screens, inventory = [], [], [], [], []
    for entry in campaign['jobs']:
        base = root / entry['path']; output = base / 'output'
        done = read(output / 'completed.json') if (output / 'completed.json').exists() else {}
        if done.get('status') != 'complete':
            inventory.append({'seed': entry['seed'], 'included': False, 'reason': 'not complete'})
            continue
        m = read(output / 'manifest.json')
        threshold = m['config']['quality_threshold']
        rr = records(output)
        if not (output / 'exploration.sqlite').exists():
            raise ValueError(f'Missing completed-run database: {output}')
        evaluator = None
        if rpa_screen and m['task'] != 'rpa':
            raise ValueError('--rpa-screen is only applicable to RPA')
        if any(r.get('region') is None and r['loss'] < threshold for r in rr) or rpa_screen:
            from RL4CRN.search_exploration.evaluation import build
            from RL4CRN.search_exploration.archive import descriptor
            evaluator, _, _ = build(m['task'], m['task_config'], m['config']['rate_min'], m['config']['rate_max'])
            library = {r.ID: r for r in evaluator.library.reactions}
            for row in rr:
                if row.get('region') is None and row['loss'] < threshold:
                    row['region'] = descriptor(row['reaction_ids'], library, evaluator.crn_template.output_labels,
                                               m['config']['descriptor'], row['topology_hash'])
        topologies, categories = summarize(rr, threshold)
        best = min((r['loss'] for r in rr), default=None)
        history = list(jsonlines(output / 'metrics.jsonl'))
        for h in history:
            if int(h['epoch']) >= done.get('epochs', 0):
                continue  # terminal draining belongs at T, not in the epoch curve
            epoch = int(h['epoch']) + 1
            observed = [r for r in rr if r['_checkpoint'] is not None and r['_checkpoint'] <= epoch]
            tops, cats = summarize(observed, threshold)
            numerical = h.get('Archive/best_loss')
            direct = [r['loss'] for r in observed if r['_record_stream'] == 'direct_llm']
            values = direct + ([numerical] if isinstance(numerical, (int, float)) and math.isfinite(numerical) else [])
            curves.append(dict(seed=entry['seed'], completed_epochs=epoch,
                numerical_best_loss=numerical, system_best_loss=min(values) if values else None,
                qualifying_topologies=len(tops), structural_categories=len(cats),
                charged_evaluations=h.get('Compute/expensive_evaluations')))
        endpoints.append(dict(seed=entry['seed'], best_loss=best, qualifying_topologies=len(topologies),
            structural_categories=len(categories), epochs=done.get('epochs'),
            charged_evaluations=done.get('counters', {}).get('expensive_evaluations'),
            threshold=threshold, records_without_epoch=sum(r['_checkpoint'] is None for r in rr)))
        for item in topologies.values():
            memberships.append(dict(seed=entry['seed'], **{k: sorted(v) if isinstance(v, set) else v for k, v in item.items()}))
        if rpa_screen:
            from RL4CRN.utils.results_database import serialize_crn
            from comparisons.search_exploration.analyze_rpa_mechanisms import integral_balance
            from comparisons.search_exploration.analyze_rpa_communication_families import antithetic_motif, aif_like
            fixed = json.loads(serialize_crn(evaluator.crn_template)['parameters_json'])
            screened = {}
            # Motif is structural; balance is rate-specific. Retain any positive
            # witness per topology rather than testing only its best-loss vector.
            for row in rr:
                if row['loss'] >= threshold:
                    continue
                reactions = list(fixed)
                for rid, rate in zip(row['reaction_ids'], row['rates']):
                    reaction = library[rid]
                    reactions.append({'structure': {'type': type(reaction).__name__, 'reactants': reaction.reactant_labels,
                        'products': reaction.product_labels, 'inputs': reaction.input_channels}, 'parameters': [rate]})
                balance = integral_balance(reactions)
                key = tuple(sorted(row['reaction_ids']))
                values = screened.setdefault(key, dict(motif=False, restricted_balance=False, motif_and_integral_balance=False))
                values['motif'] |= antithetic_motif(reactions)
                values['restricted_balance'] |= bool(balance['restricted_exact_balance'])
                values['motif_and_integral_balance'] |= aif_like(reactions, balance)
            screens.append(dict(seed=entry['seed'], qualifying_topologies=len(screened),
                **{key: sum(row[key] for row in screened.values()) for key in ['motif', 'restricted_balance', 'motif_and_integral_balance']}))
        inventory.append(dict(seed=entry['seed'], included=True, records=len(rr)))
        print(f'Analysed seed {entry["seed"]}: {len(topologies)} qualifying topologies, {len(categories)} categories', flush=True)
    if not endpoints:
        raise ValueError('No completed seeds yet; use status to monitor active runs')
    csv_write(dest / 'endpoints.csv', endpoints); csv_write(dest / 'curves.csv', curves)
    csv_write(dest / 'rpa_screens.csv', screens)
    (dest / 'topology_provenance.json').write_text(json.dumps(memberships, indent=2) + '\n')
    (dest / 'analysis.json').write_text(json.dumps(dict(experiment=campaign['experiment'], model=campaign['model'],
        seeds=inventory, complete_seeds_only=True, portfolio='union of numerical archive discoveries and valid direct LLM proposals',
        provenance='Observed routes may overlap; these are not exclusive causal credits.',
        epochs='Archive insertion epoch + 1; direct completion epoch + 1. Unmatched records appear only at terminal endpoint.',
        terminal='Includes completion of pending work; not a matched simulation allowance.'), indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    fields = [('system_best_loss', 'Best loss'), ('qualifying_topologies', 'Qualifying topologies'), ('structural_categories', 'Structural categories')]
    endfields = ['best_loss', 'qualifying_topologies', 'structural_categories']
    for ax, (field, label), endfield in zip(axes, fields, endfields):
        grouped = defaultdict(list)
        for row in curves:
            if row[field] is not None:
                grouped[row['completed_epochs']].append(row[field])
        xx = sorted(grouped)
        if xx:
            quantiles = np.array([np.quantile(grouped[x], [.25,.5,.75]) for x in xx])
            ax.plot(xx, quantiles[:,1], color='#246b8e', linewidth=1.8)
            ax.fill_between(xx, quantiles[:,0], quantiles[:,2], color='#246b8e', alpha=.18)
        terminal = [r[endfield] for r in endpoints if r[endfield] is not None]
        if terminal:
            lo, med, hi = np.quantile(terminal, [.25,.5,.75]); x = (max(xx) if xx else 0) + max(1, (max(xx) if xx else 1)*.05)
            ax.errorbar(x, med, yerr=[[med-lo],[hi-med]], fmt='o', color='#b86a31', capsize=3)
            ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= (max(xx) if xx else 0)] + [x])
            ticks = ax.get_xticks(); ax.set_xticklabels([f'{t:g}' for t in ticks[:-1]] + ['T'])
        if field == 'system_best_loss' and all(r[field] > 0 for r in curves if r[field] is not None) and terminal and min(terminal)>0:
            ax.set_yscale('log')
        ax.set(xlabel='Completed GA batches' if campaign['experiment']=='rpa-ga' else 'Completed epochs', ylabel=label)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=.15)
    fig.savefig(dest / 'overview.pdf'); fig.savefig(dest / 'overview.png', dpi=180); plt.close(fig)
    print(f'Results: {dest} (median and interquartile range; T = final synchronized result)')
