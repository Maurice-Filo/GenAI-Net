"""Freeze existing results assets and audit current-cohort first LLM requests.

Read-only access to completed cohorts; no simulation, fitting or model calls.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'jmlr_hybrid_crn'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    generated = HERE / 'results_data'
    generated.mkdir(exist_ok=True)
    sources = {}
    figures = ('rpa_optimizer_robustness.pdf', 'canonical_counts_per_seed.pdf',
               'channels.pdf', 'logic_provenance.pdf')
    tables = ('rpa_tests.tex', 'rpa_perturbation_tests.tex', 'logic_tests.tex')
    for folder, names, target in (('figures', figures, HERE / 'figures'),
                                  ('generated', tables, generated)):
        for name in names:
            source = ORIGINAL / folder / name
            shutil.copyfile(source, target / name)
            sources[str(source)] = digest(source)
            assert digest(target / name) == sources[str(source)]
    channels = ORIGINAL / 'generated/channels.json'
    sources[str(channels)] = digest(channels)
    rows = json.loads(channels.read_text())['rows']
    audited, summaries = [], []
    for task, threshold in (('rpa', .016491168958935352), ('logic', .04271267898491677)):
        selected = sorted((r for r in rows if r['task'] == task and r['arm'] == 'none'), key=lambda r: r['seed'])
        assert [r['seed'] for r in selected] == list(range(1, 21))
        for run in selected:
            bank = Path(run['llm_bank']) if task == 'rpa' else Path(run['folder']) / 'direct_llm_proposals.jsonl'
            manifest_path = bank.parent / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            assert manifest['withhold_initial_hof'] and manifest['model'] == 'deepseek-v4-flash'
            assert manifest['communication_mode'] == ('llm-to-rl' if task == 'rpa' else 'none')
            assert manifest['config']['quality_threshold'] == threshold
            entries = [json.loads(line) for line in bank.read_text().splitlines()]
            first = [r for r in entries if r['launched_epoch'] == 0]
            assert len(first) == 10 and sorted(r['candidate_index'] for r in first) == list(range(10))
            assert all(r['direct_supplied_rate_only'] for r in first)
            valid = [r['record'] for r in first if r['record'] and r['record']['simulation_complete']]
            best = min((r['loss'] for r in valid), default=float('inf'))
            audited.append({'task': task, 'seed': run['seed'], 'batch_size': len(first),
                'complete_proposals': len(valid), 'best_loss': best if np.isfinite(best) else None,
                'threshold': threshold, 'qualifying_proposals': sum(r['loss'] < threshold for r in valid),
                'batch_qualifies': best < threshold, 'source_bank': str(bank)})
            sources[str(bank)], sources[str(manifest_path)] = digest(bank), digest(manifest_path)
        task_rows = [r for r in audited if r['task'] == task]
        summaries.append({'task': task, 'runs': len(task_rows),
            'qualifying_first_batches': sum(r['batch_qualifies'] for r in task_rows),
            'qualifying_proposals': sum(r['qualifying_proposals'] for r in task_rows),
            'median_first_batch_best': float(np.median([r['best_loss'] for r in task_rows]))})
    with (generated / 'first_request_audit.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)
    (generated / 'first_request_summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    (generated / 'source_hashes.json').write_text(json.dumps(sources, indent=2) + '\n')
    for row in summaries:
        print(row)
    print('Copied figures/tables byte-for-byte; 40 initial batches audited. No new evaluations.')


if __name__ == '__main__':
    main()
