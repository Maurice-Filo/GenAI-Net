import json
import sqlite3

import pytest

from comparisons.search_exploration.analyze_logic5_directional_components import best_rl_events


def test_rl_minimum_excludes_llm_ga_and_failed_solves_and_splits_fitting():
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE designs(key TEXT PRIMARY KEY,payload TEXT)')
        db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT)')
        for i,(source,stream,loss,complete) in enumerate([
            ('llm','search',.001,True), ('ga','search',.002,True),
            ('rl','search',.04,True), ('rl','optimization',.02,True),
            ('rl','search',.0001,False)]):
            # Final design metadata may differ from the event's source.
            db.execute('INSERT INTO designs VALUES (?,?)',(str(i),json.dumps({'source':'llm','simulation_complete':complete})))
            db.execute('INSERT INTO events VALUES (?,?,?)',(i,'archive_insertion',json.dumps({
                'source':source,'stream':stream,'loss':loss,'candidate_hash':str(i)})))
        result=best_rl_events(db)
        assert result['rl_sampled_loss'] == .04
        assert result['rl_with_fit_loss'] == .02
        assert result['rl_sampled_loss_event']['source'] == 'rl'


def test_missing_rl_data_are_not_imputed():
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE designs(key TEXT PRIMARY KEY,payload TEXT)')
        db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT)')
        with pytest.raises(ValueError,match='No complete RL event'):
            best_rl_events(db)
