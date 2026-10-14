"""Count actual recorded Qwen archive admissions and explicit numerical lineages.

Read-only audit of completed runs. Admission means archive_insertion.admitted=true,
not merely simulation validity or an attempted Hall-of-Fame.add call. This flag
can also report an identical incumbent; unique candidate counts are given too.
"""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from collections import defaultdict

ROOT = Path('/local1/rossin/search-exploration/qwen38-matched-rpa-logic-20seed-20261006-r2')
OUT = Path(__file__).resolve().parent


def audit(job):
    run = Path(job['output'])
    manifest = json.loads((run / 'manifest.json').read_text())
    assert json.loads((run / 'completed.json').read_text())['status'] == 'complete'
    threshold, epochs = manifest['config']['quality_threshold'], manifest['config']['epochs']
    # Confirm that the code defining the counts is the code used in this run.
    frozen = ROOT / job['task'] / 'source'
    for name in ['RL4CRN/search_streams/engine.py', 'RL4CRN/search_exploration/archive.py']:
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == manifest['new_source'][name]
    history = json.loads((run / 'llm_history.json').read_text())
    row = dict(task=job['task'], seed=job['seed'], requested_batches=len(history),
        failed_batches=sum(bool(h.get('error')) for h in history),
        returned=sum(h.get('returned') or 0 for h in history),
        valid=sum(h.get('valid') or 0 for h in history))
    groups = defaultdict(list)
    cache = {}
    events = []
    with sqlite3.connect(f'file:{run}/exploration.sqlite?mode=ro&immutable=1', uri=True) as db:
        query = """SELECT id,payload FROM events WHERE kind='archive_insertion'
          AND (json_extract(payload,'$.origin')='llm' OR json_extract(payload,'$.source')='llm') ORDER BY id"""
        for event_id, payload in db.execute(query):
            e = json.loads(payload)
            key = e['candidate_hash']
            if key not in cache:
                found = db.execute('SELECT payload FROM designs WHERE key=?', (key,)).fetchone()
                assert found is not None, key
                d = json.loads(found[0])
                cache[key] = dict(topology_hash=d['topology_hash'], reaction_ids=d['reaction_ids'],
                                  simulation_complete=d['simulation_complete'], all_rows_correct=d['all_rows_correct'])
            d = cache[key]
            e.update(event_id=event_id, topology_hash=d['topology_hash'], reaction_ids=d['reaction_ids'])
            e['qualifying'] = bool(d['simulation_complete'] and e['loss'] < threshold and (
                not manifest['config']['require_all_rows'] or d['all_rows_correct'] is True))
            e['during_training'] = e['epoch'] < epochs
            if e['source'] == 'llm' and e['stream'] != 'optimization':
                kind = 'direct'
            elif e['source'] == 'llm' and e['stream'] == 'optimization':
                kind = 'llm_adagrad'
            else:
                kind = 'genetic_descendant'
                assert e['origin'] == 'llm'
            e['group'] = kind
            groups[kind].append(e)
            events.append(dict(task=job['task'], seed=job['seed'], **e))
    assert len(groups['direct']) == row['valid'], (job['name'], len(groups['direct']), row['valid'])
    for kind in ['direct', 'llm_adagrad', 'genetic_descendant']:
        entries = groups[kind]
        admitted = [e for e in entries if e['admitted']]
        row[kind + '_offered'] = len(entries)
        row[kind + '_offered_qualifying'] = sum(e['qualifying'] for e in entries)
        row[kind + '_admitted'] = len(admitted)
        row[kind + '_admitted_unique_candidates'] = len({e['candidate_hash'] for e in admitted})
        row[kind + '_admitted_unique_topologies'] = len({e['topology_hash'] for e in admitted})
        row[kind + '_admitted_qualifying'] = sum(e['qualifying'] for e in admitted)
        row[kind + '_admitted_qualifying_topologies'] = len({e['topology_hash'] for e in admitted if e['qualifying']})
        row[kind + '_admitted_training'] = sum(e['during_training'] for e in admitted)
        row[kind + '_admitted_training_qualifying'] = sum(e['during_training'] and e['qualifying'] for e in admitted)
        row[kind + '_admitted_after_training'] = sum(not e['during_training'] for e in admitted)
        row[kind + '_positive_region_gain_events'] = sum(e['gain'] > 0 for e in entries)
    for group, selected in [('direct_or_adagrad', groups['direct'] + groups['llm_adagrad']), ('llm_lineage', events)]:
        accepted = [e for e in selected if e['admitted']]
        row[group + '_qualifying_topologies'] = len({e['topology_hash'] for e in accepted if e['qualifying']})
        row[group + '_training_qualifying_topologies'] = len({e['topology_hash'] for e in accepted if e['qualifying'] and e['during_training']})
    return row, events


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    rows, events = [], []
    for job in plan['jobs']:
        row, detail = audit(job)
        rows.append(row); events.extend(detail)
        print(job['name'], 'direct', row['direct_admitted'], '/', row['valid'], flush=True)
    summary = {}
    for task in ('rpa', 'logic'):
        group = [r for r in rows if r['task'] == task]
        totals = {k: sum(r[k] for r in group) for k in group[0] if k not in ('task', 'seed')}
        totals['runs'] = len(group)
        for kind in ['direct', 'llm_adagrad', 'genetic_descendant']:
            totals[kind + '_runs_with_admissions'] = sum(r[kind + '_admitted'] > 0 for r in group)
            totals[kind + '_runs_with_training_admissions'] = sum(r[kind + '_admitted_training'] > 0 for r in group)
            totals[kind + '_runs_with_qualifying_admissions'] = sum(r[kind + '_admitted_qualifying'] > 0 for r in group)
        summary[task] = totals
    with (OUT / 'qwen_archive_insertions_by_seed.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / 'qwen_archive_insertion_events.jsonl').open('w') as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
    report = dict(campaign=str(ROOT), definitions={
        'admission': 'archive_insertion.admitted=true: the candidate is the retained topology representative immediately after insertion processing. This is not the top-30 Hall of Fame membership count.',
        'counts': 'Admission events; deduplicated candidate/topology counts are also supplied. Cohort unique counts sum within-run counts, not a cross-seed union.',
        'during_training': 'Recorded merge epoch <100. Epoch 100 arrivals cannot affect completed policy updates.',
        'qualifying': 'Simulation complete and loss below the task reference threshold (not a robustness or mechanistic guarantee).',
        'llm_adagrad': 'source=llm and stream=optimization: numerical parameter refinement of an LLM-source design.',
        'genetic_descendant': 'Other emissions with explicit origin=llm ancestry; excludes untraceable policy-level imitation effects.',
        'retention': 'An admission is not a guarantee of final retention, a novel biochemical mechanism, or actual sampling for imitation.',
    }, summary=summary)
    (OUT / 'qwen_archive_insertions_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
