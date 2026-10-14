"""Audit region semantics and frozen campaign sources without rerunning search.

Run from the repository root with PYTHONPATH=. .venv/bin/python.
Outputs region_descriptor_audit_checks.json alongside this script.
"""
from collections import Counter
from dataclasses import replace
import ast
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np

from RL4CRN.search_exploration.archive import descriptor
from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_streams.archive import StreamDiscoveryArchive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class Store:
    def __init__(self):
        self.visits = Counter()

    def count(self, key, increment=0):
        self.visits[key] += increment
        return self.visits[key]

    def counts(self, keys):
        return {k: self.visits[k] for k in keys}

    def save_design(self, record):
        pass


class Reaction:
    def __init__(self, reactants, products):
        self.reactant_labels, self.product_labels = reactants, products

    def get_stoichiometry_dict(self):
        a, b = Counter(self.reactant_labels), Counter(self.product_labels)
        return {s: b[s] - a[s] for s in a.keys() | b.keys()}


def record(key, region, loss, complete=True):
    return dict(topology_hash=key, candidate_hash=key + '-candidate', region=region,
                loss=loss, simulation_complete=complete, qualified=complete and loss < .1,
                reaction_ids=[1, 2], rates=[1., 1.])


def check_semantics():
    cfg = ExplorationConfig(quality_threshold=.1)
    library = {0: Reaction(['A', 'B'], ['Y']), 1: Reaction(['Y'], []),
               2: Reaction(['A', 'A'], ['Y']), 3: Reaction(['B', 'Y'], []),
               4: Reaction(['A'], ['A', 'A'])}
    d1 = descriptor([0, 1], library, ['Y'], 'reaction_roles_v1', 'one')
    d2 = descriptor([2, 3], library, ['Y'], 'reaction_roles_v1', 'two')
    assert d1 == d2 == '1:1:0:0:1:3'
    assert descriptor([4], library, ['Y'], 'reaction_roles_v1', '') == '0:0:1:0:0:1'
    archive, store = StreamDiscoveryArchive(cfg), Store()
    first = record('first', 'same-region', .02)
    gain1, novelty1 = archive.add(first, store)
    gain2, novelty2 = archive.add(record('second', 'same-region', .02), store)
    assert math.isclose(gain1, 1.2) and gain2 == 0
    assert math.isclose(novelty1, math.exp(-.2))
    assert math.isclose(novelty2, math.exp(-.2) / math.sqrt(2))
    failed = archive.add(record('failed', 'same-region', 1e4, False), store)
    assert failed == (0., 0.) and store.count('region:same-region') == 3
    repeated = archive.add(first, store)
    assert repeated[0] == 0 and store.count('region:same-region') == 4
    assert math.isclose(repeated[1], math.exp(-.2) / 2)

    # Record the actual probabilities passed by the production parent sampler.
    class Capture:
        def choice(self, values, p=None):
            self.p = list(p)
            return 0

        def integers(self, size):
            return 0

    parents, counts = StreamDiscoveryArchive(cfg), Store()
    parents.add(record('a', 'A', .01), counts)
    parents.add(record('b', 'B', .02), counts)
    parents.add(record('c', 'B', .03), counts)
    counts.visits.update({'region:A': 0, 'region:B': 7})  # totals A=1, B=9
    rng = Capture()
    parents.parent_sampler(counts)(rng)
    weights = np.array([1 / math.sqrt(2), 1 / math.sqrt(10)])
    np.testing.assert_allclose(rng.p, weights / weights.sum())
    draws = parents.imitation(np.random.default_rng(7))
    assert len(draws) == 3

    # A weaker regional representative survives ahead of a stronger non-representative.
    bounded = StreamDiscoveryArchive(replace(cfg, archive_capacity=2, sil_top_k=2))
    ledger = Store()
    for r in [record('A-best', 'A', .01), record('A-other', 'A', .02), record('B-best', 'B', .09)]:
        bounded.add(r, ledger)
    assert set(bounded.entries) == {'A-best', 'B-best'}
    # Elite eligibility is ranked loss, not a requirement to pass the threshold.
    unqualified = StreamDiscoveryArchive(cfg)
    unqualified.add(record('not-qualified', 'Q', .2), Store())
    assert unqualified.imitation(np.random.default_rng(1))[0]['qualified'] is False
    return dict(descriptor_collision=d1, overlap_includes_autocatalysis=True,
                first_reward=gain1 + cfg.novelty_weight * novelty1,
                same_loss_second_reward=gain2 + cfg.novelty_weight * novelty2,
                failures_and_repeats_increment_visits=True,
                parent_region_probabilities=dict(zip(['A', 'B'], rng.p)),
                regional_retention_example=sorted(bounded.entries),
                elite_threshold_not_required=True)


def schedule_ast(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JuliaStreamTrainer')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_schedule')
    return ast.dump(fn, include_attributes=False)


def check_manifests():
    rows = json.loads((ROOT / 'paper/jmlr_hybrid_crn/generated/channels.json').read_text())['rows']
    folders = sorted({r['folder'] for r in rows})
    files = ['RL4CRN/search_exploration/archive.py', 'RL4CRN/search_streams/archive.py',
             'RL4CRN/search_exploration/policy.py', 'RL4CRN/search_exploration/engine.py',
             'RL4CRN/search_streams/engine.py', 'RL4CRN/search_streams/julia_stream.py']
    keys = ['descriptor', 'sil_region_balanced', 'sil_top_k', 'sil_batch_size',
            'archive_capacity', 'novelty_weight', 'quality_improvement_weight',
            'imitation_weight', 'entropy_weight', 'require_all_rows']
    result = dict(folders=len(folders), configurations={}, source_checks={},
                  differing_julia_sources=[], missing=[])
    for folder in folders:
        path = Path(folder) / 'manifest.json'
        if not path.exists():
            result['missing'].append(str(path))
            continue
        data = json.loads(path.read_text())
        for key in keys:
            value = json.dumps(data['config'].get(key), sort_keys=True)
            counts = result['configurations'].setdefault(key, {})
            counts[value] = counts.get(value, 0) + 1
        hashes = data.get('new_source', data.get('source_hashes', {}))
        for name in files:
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            state = 'match' if hashes.get(name) == actual else 'different'
            counts = result['source_checks'].setdefault(name, {})
            counts[state] = counts.get(state, 0) + 1
            if state == 'different':
                assert name.endswith('julia_stream.py'), (folder, name)
                with tarfile.open(Path(folder) / 'source.tar.gz') as archive:
                    member = next(n for n in archive.getnames() if n.endswith(name))
                    raw = archive.extractfile(member).read()
                assert hashlib.sha256(raw).hexdigest() == hashes[name]
                same = schedule_ast(raw.decode()) == schedule_ast((ROOT / name).read_text())
                assert same
                result['differing_julia_sources'].append(dict(folder=folder,
                    frozen_sha256=hashes[name], scheduler_ast_matches=same))
    assert not result['missing']
    return result


if __name__ == '__main__':
    result = dict(semantics=check_semantics(), campaigns=check_manifests())
    out = HERE / 'region_descriptor_audit_checks.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['semantics'], indent=2))
    print(f"Checked {result['campaigns']['folders']} run manifests; results: {out}")
