"""Portable campaign preparation/routing and honest fresh-result accounting."""
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'reproduce' / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load('search')
analysis = load('search_analysis')


def test_seed_selection():
    assert cli.seeds('1,3-5,3') == [1,3,4,5]
    for invalid in ['0', '21', '5-3', '', '1-2-3']:
        with pytest.raises(argparse.ArgumentTypeError):
            cli.seeds(invalid)


def test_model_configuration_stays_in_harness(tmp_path):
    patch = tmp_path/'cordis.patch.yml'
    value = '- id: agent-default-model\n  config:\n    provider: private-server\n    model: Qwen/model\n'
    patch.write_text(value)
    assert cli.model_route('private-server/Qwen/model', tmp_path) == ('private-server', 'Qwen/model')
    assert patch.read_text() == value
    patch.write_text(value.replace('Qwen/model','my-model'))
    assert cli.model_route('my-model',tmp_path)==('private-server','my-model')
    with pytest.raises(ValueError, match='uniquely resolve'):
        cli.model_route('unknown-model',tmp_path)


def test_model_ambiguity_is_not_silently_resolved(tmp_path):
    (tmp_path/'cordis.patch.yml').write_text('''- id: llm-pi-ai
  config:
    providers:
      one:
        models: [{id: alias}]
      two:
        models: [{id: alias}]
''')
    with pytest.raises(ValueError, match='uniquely'):
        cli.model_route('alias', tmp_path)


def test_transport_overrides_only_routing():
    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    load('search_transport').install(Client,'configured','chosen-model')
    client = Client(provider='hardcoded', model='old', openai_compatible_base_url='http://old:8000/v1',
                    timeout_seconds=3600, system_prompt='preserve scientific instructions')
    assert client.kwargs == dict(provider='configured', model='chosen-model', openai_compatible_base_url=None,
                                timeout_seconds=3600, system_prompt='preserve scientific instructions')


def test_real_client_fork_keeps_routing_without_reconfiguring_provider(tmp_path, monkeypatch):
    from RL4CRN.llm.harness_client import HarnessLLMClient
    monkeypatch.setattr(HarnessLLMClient, '__init__', HarnessLLMClient.__init__)
    load('search_transport').install(HarnessLLMClient, 'user-provider', 'user-model')
    client = HarnessLLMClient(workspace_root=tmp_path/'workspace', dsh_home=tmp_path/'harness',
                              command=['not-executed'], openai_compatible_base_url='http://localhost:9000/v1')
    fork = client.fork()
    assert (fork.provider, fork.model, fork.openai_compatible_base_url)==('user-provider','user-model',None)
    patch=fork._profile_patch_text()
    assert 'user-model' in patch and 'user-provider' in patch
    assert 'baseURL:' not in patch and 'apiKeyEnv:' not in patch


def test_manifest_identity_uses_effective_model_but_candidate_keys_do_not_change():
    store=SimpleNamespace(fingerprint=lambda value:json.dumps(value,sort_keys=True))
    transport=load('search_transport')
    transport.install_manifest_metadata(store,'configured','chosen')
    candidate={'reaction_ids':[1], 'rates':[.1]}
    before=dict(candidate)
    assert store.fingerprint(candidate)==json.dumps(before,sort_keys=True) and candidate==before
    manifest=dict(schema='search', config={'seed':1}, harness=True, model='old',provider='old-provider',
                  openai_compatible_base_url='http://old/v1')
    identity=store.fingerprint(manifest)
    assert json.loads(identity)['model']=='chosen'
    assert manifest['reference_transport']['model']=='old'
    assert 'openai_compatible_base_url' not in manifest
    assert store.fingerprint(manifest)==identity


def test_unsafe_archives_rejected(tmp_path):
    archive = tmp_path/'bad.tar.gz'; destination=tmp_path/'target'; destination.mkdir()
    with tarfile.open(archive,'w:gz') as stream:
        member=tarfile.TarInfo('../outside');member.type=tarfile.DIRTYPE;stream.addfile(member)
    with pytest.raises(ValueError, match='Unsafe'):
        cli.unpack(archive,destination)
    assert not (tmp_path/'outside').exists()


def test_catalog_does_not_invent_independent_rpa_arm():
    assert 'rpa-independent' not in cli.EXPERIMENTS
    assert cli.EXPERIMENTS['logic-independent'][1]=='logic_none'
    assert len(cli.EXPERIMENTS)==19


def test_cohort_selection_is_exact(tmp_path):
    (tmp_path/'manifest').mkdir()
    (tmp_path/'manifest/cohorts.csv').write_text('selection,group,seed,run\nLLM resource accounting,logic_full,1,abc\nOther,logic_full,1,wrong\n')
    assert cli.select_runs(tmp_path,'logic-full') == {1:'abc'}


def test_counts_include_distinct_topologies_not_parameterizations():
    rows=[dict(simulation_complete=True, loss=.01, reaction_ids=[1,2],region='a',origin='llm',source='llm'),
          dict(simulation_complete=True, loss=.02, reaction_ids=[2,1],region='a',origin='rl',source='optimization'),
          dict(simulation_complete=True, loss=.03, reaction_ids=[3,4],region='b'),
          dict(simulation_complete=False, loss=.001, reaction_ids=[9],region='c'),
          dict(simulation_complete=True, loss=.1, reaction_ids=[8],region='d')]
    tops,cats=analysis.summarize(rows,.1)
    assert len(tops)==2 and cats=={'a','b'}
    assert len(tops[(1,2)]['observed_routes'])==2


def test_archive_and_independent_llm_records_are_both_preserved(tmp_path):
    row=dict(candidate_hash='one',reaction_ids=[1,2],rates=[1,1],simulation_complete=True,loss=.01)
    with sqlite3.connect(tmp_path/'exploration.sqlite') as db:
        db.executescript('CREATE TABLE events(id INTEGER,kind TEXT,payload TEXT); CREATE TABLE designs(key TEXT,payload TEXT);')
        db.execute('INSERT INTO designs VALUES (?,?)',('one',json.dumps(row)))
        db.execute('INSERT INTO events VALUES (1,?,?)',('archive_insertion',json.dumps(dict(candidate_hash='one',epoch=4))))
    (tmp_path/'direct_llm_proposals.jsonl').write_text(json.dumps(dict(record={**row,'reaction_ids':[3,4]},completed_epoch=5))+'\n')
    rows=analysis.records(tmp_path)
    assert [r['_checkpoint'] for r in rows]==[5,6]
    assert len(analysis.summarize(rows,.1)[0])==2


def test_tampered_job_is_detected(tmp_path):
    (tmp_path/'launcher').mkdir(); jobdir=tmp_path/'jobs/seed-01';jobdir.mkdir(parents=True)
    cli.write(jobdir/'job.json',dict(source_hashes={},input_hashes={}))
    cli.write(tmp_path/'campaign.json',dict(launcher_hashes={},jobs=[dict(seed=1,path='jobs/seed-01',job_sha256='wrong')],
                                          model=None,provider=None))
    with pytest.raises(ValueError,match='Preflight'):
        cli.check(argparse.Namespace(campaign=tmp_path,files_only=True,python=None))


def test_completed_seed_is_not_reexecuted(tmp_path, monkeypatch):
    (tmp_path/'jobs/seed-01').mkdir(parents=True)
    cli.write(tmp_path/'campaign.json',dict(python=sys.executable,parallel_runs=1,compaction=False,
                jobs=[dict(seed=1,path='jobs/seed-01',state='complete')]))
    monkeypatch.setattr(cli,'check',lambda _: None)
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k: pytest.fail('Must not reexecute completed seed'))
    cli.run(argparse.Namespace(campaign=tmp_path,python=None,parallel_runs=None))


def test_runner_continues_after_failed_seed(tmp_path, monkeypatch):
    (tmp_path/'launcher').mkdir()
    (tmp_path/'launcher/search_worker.py').write_text('''import json,sys
from pathlib import Path
p=Path(sys.argv[1]).parent
if p.name=='seed-01': raise SystemExit(3)
o=p/'output';o.mkdir()
(o/'completed.json').write_text(json.dumps({'status':'complete'}))
''')
    jobs=[]
    for seed in [1,2]:
        base=tmp_path/'jobs'/f'seed-{seed:02d}';(base/'source').mkdir(parents=True)
        cli.write(base/'job.json',{})
        jobs.append(dict(seed=seed,path=str(base.relative_to(tmp_path)),state='pending'))
    cli.write(tmp_path/'campaign.json',dict(python=sys.executable,parallel_runs=1,compaction=False,jobs=jobs))
    monkeypatch.setattr(cli,'check',lambda _:None)
    with pytest.raises(ValueError, match='Some runs'):
        cli.run(argparse.Namespace(campaign=tmp_path,python=None,parallel_runs=None))
    states=[r['state'] for r in cli.read(tmp_path/'campaign.json')['jobs']]
    assert states==['failed','complete']
