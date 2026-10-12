import json
import sqlite3

import pytest

from comparisons.search_exploration.plot_task_channel_effects import contrasts, direct_topologies


def test_direct_llm_yield_filters_and_deduplicates():
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE designs(key TEXT PRIMARY KEY,payload TEXT)')
        db.execute('CREATE TABLE events(kind TEXT,payload TEXT)')
        for i, (topology, source, stream, loss, complete) in enumerate([
            ('a', 'llm', 'search', .01, True),
            ('a', 'llm', 'search', .02, True),
            ('b', 'llm', 'optimization', .001, True),
            ('c', 'rl', 'search', .001, True),
            ('d', 'llm', 'search', .001, False),
            ('e', 'llm', 'search', .03, True),
            ('f', 'llm', 'search', -1, True),
        ]):
            db.execute('INSERT INTO designs VALUES (?,?)', (str(i), json.dumps({
                'topology_hash': topology, 'simulation_complete': complete})))
            db.execute('INSERT INTO events VALUES (?,?)', ('archive_insertion', json.dumps({
                'source': source, 'stream': stream, 'loss': loss, 'candidate_hash': str(i)})))
        assert direct_topologies(db, .03) == {'a'}


def test_channel_effects_orient_improvement_and_intervals():
    rows = []
    for task in ('logic', 'rpa'):
        for arm in ('none', 'llm-to-rl', 'rl-to-llm', 'full'):
            insertion = arm in ('llm-to-rl', 'full')
            outward = arm in ('rl-to-llm', 'full')
            for seed in range(1, 21):
                rows.append(dict(task=task, arm=arm, seed=seed,
                                 llm_yield=seed + (3 if outward else 0),
                                 rl_with_fit_loss=seed * (.5 if insertion else 1),
                                 rl_sampled_loss=seed))
    results = contrasts(rows, [])
    assert len(results) == 12
    for result in results:
        expected = {'llm_yield': 3, 'rl_with_fit_loss': 50, 'rl_sampled_loss': 0}[result['metric']]
        assert result['effect'] == pytest.approx(expected)
        assert result['ci95'] == pytest.approx([expected, expected])
        if result['task'] == 'rpa':
            assert result['holm_p_30_logic'] is None
