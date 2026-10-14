"""Freeze completed RPA timeout-follow-up trajectories; no simulations or inference."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path('/local1/rossin/search-exploration/qwen38-rpa-timeout3600-20seed-20261007')
OUT = Path(__file__).with_name('qwen_rpa_timeout3600_snapshot.json')


def main():
    hashes = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)

    plan = read(ROOT / 'plan.json')
    read(ROOT / 'finished.json')
    design = read(ROOT / 'comparison_design.json')
    execution = read(ROOT / 'execution_settings.json')
    runs, histories, at_allowance = [], [], []
    for job in sorted(plan['jobs'], key=lambda j: j['seed']):
        assert job['task'] == 'rpa'
        folder = Path(job['output'])
        done = read(folder / 'completed.json')
        assert done['status'] == 'complete' and done['epochs'] == 100
        assert done['termination'] == 'epochs_and_all_streams_drained'
        manifest = read(folder / 'manifest.json')
        assert manifest['config']['quality_threshold'] == 0.016491168958935352
        records = read(folder / 'metrics.jsonl', lines=True)
        checkpoints = sorted({m['epoch']: m for m in records}.values(), key=lambda m: m['epoch'])
        assert [m['epoch'] for m in checkpoints] == list(range(101))
        cost = [m['Compute/expensive_evaluations'] for m in checkpoints]
        loss = [m['Archive/best_loss'] for m in checkpoints]
        assert all(a <= b for a, b in zip(cost, cost[1:]))
        assert all(0 < b <= a + 1e-10 for a, b in zip(loss, loss[1:]))
        assert cost[0] <= 102400 <= cost[-1]
        assert cost[-1] == done['counters']['expensive_evaluations']
        assert loss[-1] == done['metrics']['Archive/best_loss']
        at_allowance.append([v for c, v in zip(cost, loss) if c <= 102400][-1])
        history = read(folder / 'llm_history.json')
        histories.extend(history)
        runs.append(dict(task='rpa', arm='qwen_60min', seed=job['seed'],
                         folder=str(folder), status='complete', role_timeout_seconds=3600,
                         epoch=[m['epoch'] + 1 if m['epoch'] < 100 else 101 for m in checkpoints],
                         post_training_drain=[m['epoch'] == 100 for m in checkpoints],
                         cost=cost, loss=loss))
    assert [r['seed'] for r in runs] == list(range(1, 21))
    errors = [h for h in histories if h.get('error')]
    summary = dict(runs=20, successful_batches=len(histories)-len(errors),
                   attempted_batches=len(histories), failed_batches=len(errors),
                   returned=sum(h.get('returned') or 0 for h in histories),
                   valid=sum(h.get('valid') or 0 for h in histories),
                   median_at_102400=median(at_allowance),
                   median_final=median(r['loss'][-1] for r in runs))
    OUT.write_text(json.dumps(dict(campaign=str(ROOT), design=design, execution=execution,
                                  source_hashes=hashes, runs=runs, summary=summary,
                                  alignment='Last recorded best loss at or below charged allowance; no extrapolation.'),
                              indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
