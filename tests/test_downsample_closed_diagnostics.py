import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pytest

from comparisons.search_exploration.downsample_closed_diagnostics import transformed_blob, migrate
from comparisons.search_exploration.analyze_rpa_mechanisms import cached_function


def info():
    times=np.linspace(0,100,1000)
    inputs=np.array([(a,b) for a in [.5,1,1.5] for b in [.5,1,1.5]])
    y=np.array([[a+np.exp(-times)] for a,b in inputs])
    return {'time_horizon':times.tolist(),'outputs':y.tolist(),
            'trajectories':np.repeat(y,3,axis=1).tolist(),'inputs':inputs.tolist(),
            'setpoint':inputs[:,0,None].tolist(),'reward':.42,'notes':'preserve me'}


def blob():return zlib.compress(json.dumps(info()).encode())


def test_retains_exact_time_samples_scores_and_rpa_drift():
    original=info();packed,kind=transformed_blob(blob())
    sampled=json.loads(zlib.decompress(packed))
    assert kind=='downsampled' and len(sampled['time_horizon'])==101
    assert sampled['reward']==original['reward'] and sampled['notes']==original['notes']
    np.testing.assert_array_equal(np.asarray(sampled['trajectories'])[...,-1],np.asarray(original['trajectories'])[...,-1])
    assert cached_function(sampled)==cached_function(original)
    assert len(packed)<len(blob())/3
    with pytest.raises(ValueError,match='Already sampled'):transformed_blob(packed)


def make_run(tmp_path):
    root=tmp_path/'run';root.mkdir()
    for name in ('completed','status'):(root/f'{name}.json').write_text(json.dumps({'status':'complete','epochs':100}))
    (root/'manifest.json').write_text(json.dumps({'identity':'fixture','config':{'quality_threshold':.05}}))
    con=sqlite3.connect(root/'exploration.sqlite')
    con.executescript('''CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE evaluations(key TEXT PRIMARY KEY,payload BLOB);
      CREATE TABLE evaluation_summaries(key TEXT PRIMARY KEY,payload TEXT);
      CREATE TABLE diagnostics(key TEXT PRIMARY KEY,payload BLOB);
      CREATE TABLE designs(key TEXT PRIMARY KEY,payload TEXT);
      CREATE TABLE events(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT,created REAL);''')
    con.execute('INSERT INTO meta VALUES (?,?)',('identity','fixture'))
    con.execute('INSERT INTO events VALUES (?,?,?,?)',(7,'provenance','unchanged',1.25))
    for key,topo,loss in [('good','A',.01),('same_topology_bad_rates','A',2.),('bad','B',2.)]:
        r={'candidate_hash':key,'topology_hash':topo,'loss':loss,'rates':[1.,2.],'reaction_ids':[1,2],'parent_hash':'ancestor'}
        con.execute('INSERT INTO evaluation_summaries VALUES (?,?)',(key,json.dumps(r)))
        con.execute('INSERT INTO designs VALUES (?,?)',(key,json.dumps(r)))
        con.execute('INSERT INTO diagnostics VALUES (?,?)',(key,blob()))
    con.commit();con.close();return root


def test_migration_preserves_all_scalar_rows_and_good_topology_parameterizations(tmp_path):
    run=make_run(tmp_path)
    with sqlite3.connect(run/'exploration.sqlite') as c:
        original={t:c.execute(f'SELECT rowid,* FROM {t}').fetchall() for t in ['meta','evaluations','evaluation_summaries','designs','events']}
    result=migrate(run,workers=1,min_free_gib=0)
    assert result['status']=='complete'
    assert result['counts']['protected']==2 and result['counts']['downsampled']==1
    with sqlite3.connect(run/'exploration.sqlite') as c:
        for t,rows in original.items():assert c.execute(f'SELECT rowid,* FROM {t}').fetchall()==rows
        for key in ['good','same_topology_bad_rates']:
            assert c.execute('SELECT payload FROM diagnostics WHERE key=?',(key,)).fetchone()[0]==blob()
        packed=c.execute("SELECT payload FROM diagnostics WHERE key='bad'").fetchone()[0]
        assert len(json.loads(zlib.decompress(packed))['time_horizon'])==101
        assert c.execute('PRAGMA integrity_check').fetchone()==('ok',)
    assert migrate(run)['status']=='complete'


def test_refuses_live_or_uncheckpointed_runs(tmp_path):
    run=make_run(tmp_path)
    (run/'status.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError,match='completed'):migrate(run,min_free_gib=0)
    assert not (run/'diagnostic_downsampling.json').exists()


def test_failure_preserves_original_database(tmp_path):
    run=make_run(tmp_path)
    with sqlite3.connect(run/'exploration.sqlite') as c:
        d=info();d['outputs'][0][0].pop()
        c.execute("UPDATE diagnostics SET payload=? WHERE key='bad'",(zlib.compress(json.dumps(d).encode()),))
    original=(run/'exploration.sqlite').read_bytes()
    with pytest.raises(ValueError,match='Unaligned'):migrate(run,workers=1,min_free_gib=0)
    assert (run/'exploration.sqlite').read_bytes()==original
    assert json.loads((run/'diagnostic_downsampling.json').read_text())['status']=='failed_original_retained'
