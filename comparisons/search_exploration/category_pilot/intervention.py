"""Opt-in combined category/reward ablation; production search stays untouched."""
import hashlib
import math
from pathlib import Path

import numpy as np

from RL4CRN.search_exploration.archive import DiscoveryArchive
from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.search_exploration.store import canonical
from RL4CRN.search_streams.julia_stream import JuliaStreamTrainer


class SingleCategoryArchive(DiscoveryArchive):
    """Completed topology minima, loss-only capacity, uniform live draws."""
    def utility(self, record):
        return 0.  # Qualification is reporting metadata only.

    def add(self, record, store):
        record['region'] = 'single'
        loss = float(record['loss'])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('Expected a finite nonnegative canonical loss')
        store.save_design(record)
        store.count('region:single', 1)
        store.count('topology:' + record['topology_hash'], 1)
        if not record['simulation_complete']:
            return 0., 0.
        old = self.region_best.get('single')
        if old is None or loss < old['loss']:
            self.region_best['single'] = {k: record[k] for k in ('loss', 'qualified', 'topology_hash')}
        key = record['topology_hash']
        incumbent = self.entries.get(key)
        if incumbent is None or loss < incumbent['loss']:
            self.revision += 1
            self.entries[key] = {k: v for k, v in record.items() if k != 'task_info'}
            self.entries[key]['revision'] = self.revision
            if len(self.entries) > self.cfg.archive_capacity:
                # Revision/hash only resolve exact physical-loss ties.
                worst = max(self.entries, key=lambda k: (self.entries[k]['loss'],
                            -self.entries[k]['revision'], k))
                del self.entries[worst]
            self._ranked = None
        return 0., 0.

    def _eviction_priority(self, record):
        return (-record['loss'],)

    def parent_sampler(self, store):
        entries = list(self.entries.values())
        return lambda rng: entries[int(rng.integers(len(entries)))] if entries else None

    def imitation(self, rng):
        top = self.ranked(self.cfg.sil_top_k)
        if not top:
            return []
        return [top[int(i)] for i in rng.choice(len(top),
                size=min(len(top), self.cfg.sil_batch_size), replace=True)]


def negative_canonical_loss(record):
    loss = float(record['loss'])
    if not math.isfinite(loss) or loss < 0 or (not record['simulation_complete'] and loss == 0):
        raise ValueError('Missing/invalid canonical failure loss: stop without surrogate reward')
    return -loss


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PilotTrainer(JuliaStreamTrainer):
    """Identical numerical recipe with local, non-random diagnostic observation."""
    def __init__(self, *args, condition, **kwargs):
        if condition not in ('baseline', 'ablation'):
            raise ValueError('Exactly two conditions')
        self.condition = condition
        super().__init__(*args, **kwargs)
        if condition == 'ablation':
            self.discovery = SingleCategoryArchive(self.config)
        self.cohort_records = []
        atomic_json(self.output / 'initialization.json', {
            'policy_sha256': tensor_digest(self.learner), 'baseline': self.learner.baseline,
            'context': self.context(), 'policy_shapes': {k: list(v.shape) for k,v in self.learner.state_dict().items()}})
        original_update = self.learner.update
        def observed_update(proposals, rewards, context, elites):
            baseline = self.learner.baseline
            # Observe the same float32 rewards/advantages used by the unchanged learner.
            import torch
            r = torch.tensor(rewards, dtype=next(self.learner.parameters()).dtype)
            advantages = (r - baseline).tolist()
            result = original_update(proposals, rewards, context, elites)
            row = {'epoch': self.state.epoch, 'baseline_before': baseline,
                'baseline_after': self.learner.baseline, 'rewards': rewards, 'advantages': advantages,
                'context': context, 'rl_proposal_ids': [p['proposal_id'] for p in proposals],
                'elites': [e['topology_hash'] for e in elites], 'metrics': result,
                'search_records': self.cohort_records}
            with (self.output / 'learning.jsonl').open('a') as f:
                f.write(canonical(row) + '\n')
            self.cohort_records = []
            return result
        self.learner.update = observed_update

    def configure_llm_graph(self, *args, **kwargs):
        raise RuntimeError('LLM services are prohibited in this pilot')

    def _maybe_run_llm_graph(self, epoch):
        if self._llm_loop:
            raise RuntimeError('Unexpected model service configuration')

    def _record(self, record, source, proposal_id):
        reward = negative_canonical_loss(record) if self.condition == 'ablation' else None
        gain, novelty = super()._record(record, source, proposal_id)
        if reward is None:
            reward = gain + self.config.novelty_weight * novelty
        row = {k: record.get(k) for k in ('candidate_hash', 'topology_hash', 'loss',
            'simulation_complete', 'all_rows_correct', 'stream', 'origin', 'rates', 'reaction_ids')}
        row.update(source=source, proposal_id=proposal_id, epoch=self.state.epoch, reward=reward)
        self.store.event('pilot_candidate', row)
        if record.get('stream') == 'search':
            self.cohort_records.append(row)
        return (reward, 0.) if self.condition == 'ablation' else (gain, novelty)
