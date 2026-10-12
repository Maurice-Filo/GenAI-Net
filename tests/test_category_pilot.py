from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import json
import os

import numpy as np
import pytest

from comparisons.search_exploration.category_pilot.intervention import (
    SingleCategoryArchive, negative_canonical_loss, PilotTrainer, tensor_digest)
from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_streams.archive import StreamDiscoveryArchive
from RL4CRN.search_streams.julia_stream import JuliaStreamTrainer


def store():
    counts=Counter()
    def count(k,n=0):
        counts[k]+=n
        return counts[k]
    return SimpleNamespace(count=count, counts=lambda keys:{k:counts[k] for k in keys}, save_design=lambda r:None)


def record(key,loss,complete=True,region='different',qualified=False):
    return dict(topology_hash=str(key),candidate_hash=f'{key}-{loss}',region=region,
        loss=loss,qualified=qualified,simulation_complete=complete,reaction_ids=[2,1],rates=[1.,1.])


def test_one_category_loss_dedup_eviction_reporting_only():
    a=SingleCategoryArchive(ExplorationConfig(archive_capacity=2,sil_top_k=2));s=store()
    for r in [record('a',.1,qualified=False),record('b',.9,qualified=True),record('c',.2)]:
        assert a.add(r,s)==(0.,0.)
    assert set(a.entries)=={'a','c'}
    assert set(a.region_best)=={'single'}
    assert {r['region'] for r in a.entries.values()}=={'single'}
    a.add(record('a',.05),s);a.add(record('a',.08),s)
    assert a.entries['a']['loss']==.05
    a.add(record('failed',.001,complete=False),s)
    assert 'failed' not in a.entries
    restored=SingleCategoryArchive(a.cfg);restored.load_state_dict(a.state_dict())
    assert restored.ranked()==a.ranked()


@pytest.mark.parametrize('loss,complete',[(0,True),(.01,True),(15000.,True),(998.,False),(9999.,False)])
def test_reward_exact_without_threshold_bonus_or_clipping(loss,complete):
    assert negative_canonical_loss(record('a',loss,complete)) == -loss


@pytest.mark.parametrize('loss,complete',[(float('nan'),True),(float('inf'),False),(-1,True),(0,False)])
def test_invalid_canonical_reward_fails_closed(loss,complete):
    with pytest.raises(ValueError):negative_canonical_loss(record('a',loss,complete))


def test_uniform_parent_and_global_top30_elites_with_replacement():
    a=SingleCategoryArchive(ExplorationConfig());s=store()
    for i in range(40):a.add(record(i,i+1),s)
    rng=np.random.default_rng(81);parent=a.parent_sampler(s)
    counts=Counter(parent(rng)['topology_hash'] for _ in range(40000))
    assert len(counts)==40 and all(850<n<1150 for n in counts.values())
    counts=Counter();duplicates=False
    for _ in range(4000):
        batch=a.imitation(rng);assert len(batch)==16
        names=[r['topology_hash'] for r in batch]
        duplicates |= len(set(names))<16
        counts.update(names)
    assert duplicates and set(counts)==set(map(str,range(30)))
    assert all(1900<n<2400 for n in counts.values())


def test_baseline_regional_behavior_retained():
    a=StreamDiscoveryArchive(ExplorationConfig(archive_capacity=2,sil_top_k=2));s=store()
    for r in [record('a',.01,region='r1',qualified=True),record('b',.09,region='r2'),record('c',.02,region='r1',qualified=True)]:
        a.add(r,s)
    assert set(a.entries)=={'a','b'}
    fresh=StreamDiscoveryArchive(ExplorationConfig(quality_threshold=.1))
    gain,novelty=fresh.add(record('x',.02,qualified=True),store())
    assert gain+.1*novelty == pytest.approx(1.2+.1*np.exp(-.2))


def test_numerical_paths_and_policy_context_are_inherited():
    from RL4CRN.search_exploration.engine import ExplorationTrainer
    assert PilotTrainer._schedule is JuliaStreamTrainer._schedule
    assert PilotTrainer.step_epoch is JuliaStreamTrainer.step_epoch
    assert PilotTrainer.context is ExplorationTrainer.context
    t=PilotTrainer.__new__(PilotTrainer);t.config=ExplorationConfig();t.store=store()
    t.discovery=SingleCategoryArchive(t.config)
    assert t.context()==[0,0,0,0]
    t.discovery.add(record('a',.01),t.store)
    assert t.context()==pytest.approx([1/2000,1/2000,np.exp(-.01/.05),0])
    t._llm_loop=None;t._maybe_run_llm_graph(0)
    with pytest.raises(RuntimeError):t.configure_llm_graph(object())


def test_real_julia_identical_inputs_identical_refinement():
    if not os.environ.get('SYNTH_EVO_PROJECT'):pytest.skip('Julia integration needs SYNTH_EVO_PROJECT')
    from RL4CRN.parameter_optimization import JuliaOptimizer, ReactionLibrary, Reaction, Experiment
    from RL4CRN.search_streams.julia_fitting import TopologyFitter
    library=ReactionLibrary(('X',),(),(),{7:Reaction((1,),(0,))})
    ts=np.linspace(0,3,61)
    experiment=Experiment(ts,[[1.]], [[]],[[1.]],np.exp(-.7*ts)[None,None,:],loss='l2')
    with JuliaOptimizer(project=os.environ['SYNTH_EVO_PROJECT'],startup_timeout=240) as worker:
        results=[]
        for condition in ('baseline','ablation'):
            fitter=TopologyFitter(worker,library,experiment)
            first=fitter.fit({'reaction_ids':[7],'rates':[.2]},17,24,timeout=None)
            result=fitter.fit({'reaction_ids':[7],'rates':[.2]},17,24,prior=first,timeout=None)
            results.append((first,result))
        for a,b in zip(*results):
            for key in ('best_rates','best_loss','evaluations','scenario_solves','states'):
                assert a[key]==b[key]


def test_instrumented_baseline_exact_updates_and_ablation_failure_credit(tmp_path):
    import torch
    from RL4CRN.search_exploration.evaluation import build, EvaluationService, materialize
    from RL4CRN.search_exploration.session import build_session
    from RL4CRN.search_streams.store import CompactStore
    from RL4CRN.search_streams.julia_stream import JuliaRecipe
    from RL4CRN.utils.results_database import serialize_crn
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
    cfg=ExplorationConfig(epochs=2,batch_size=8,seed=3,evaluation_budget=100,plot_every=10)
    values=[]
    for condition in ('original','baseline','ablation'):
        path=tmp_path/condition;path.mkdir()
        evaluator,task,sc=build('rpa',json.loads(CONFIGS['rpa'].read_text()),.001,100.)
        sc.train.seed=cfg.seed;sc.train.batch_size=8;sc.train.n_cpus=1
        sc.policy.width=16;sc.policy.depth=1;sc.policy.deep_layer_size=16
        session=build_session(sc,task)
        db=CompactStore(path/'exploration.sqlite',100,'test',16)
        class Pool:
            slots=[None];compute_seconds=1.
            def run(self,job):
                c=job['candidate'];identity=serialize_crn(materialize(evaluator,c).state)
                complete=int(c['proposal_id'].split('proposal')[-1])%2==0
                r={k:identity[k] for k in ('topology_hash','candidate_hash')}
                r.update(loss=.2 if complete else 9999.,simulation_complete=complete,all_rows_correct=None)
                action,key=db.reserve(c,'search')
                if action=='evaluate':db.complete(key,r)
                else:r=key
                return {'best':dict(c,**r),'failures':[]}
        pool=Pool()
        opt=SimpleNamespace(compute_seconds=0.,job_estimate=1.,recipe=JuliaRecipe())
        service=EvaluationService(db,pool,[r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids],5,(.001,100.))
        cls=JuliaStreamTrainer if condition=='original' else PilotTrainer
        extra={} if condition=='original' else {'condition':condition}
        trainer=cls(session,evaluator,service,cfg,path,search_pool=pool,optimization_pool=opt,
            optimization_workers=1,queue_capacity=2,optimize_per_epoch=1,**extra)
        trainer._schedule=lambda records:0
        trainer._maybe_persist_hof=lambda *args,**kw:None
        initial=tensor_digest(trainer.learner)
        try:
            trainer.step_epoch();trainer.step_epoch()
            values.append((initial,tensor_digest(trainer.learner),trainer.discovery.ranked(),trainer.learner.baseline))
            if condition=='ablation':
                rows=[json.loads(line) for line in (path/'learning.jsonl').read_text().splitlines()]
                for row in rows:
                    byid={r['proposal_id']:r for r in row['search_records']}
                    assert row['rewards']==[-byid[i]['loss'] for i in row['rl_proposal_ids']]
                    assert all(r['simulation_complete'] for r in trainer.discovery.entries.values())
                assert any(r['reward']==-9999. for row in rows for r in row['search_records'])
                assert db.count('evaluations/llm')==0
        finally:
            trainer.close_streams();trainer.close_results_database();db.close()
    assert values[0]==values[1]
    assert values[1][0]==values[2][0]
    assert values[1][1]!=values[2][1]


def test_pilot_recipe_records_disagreements_for_both_tasks_without_other_changes():
    from comparisons.search_exploration.category_pilot.prepare import pilot_julia_recipe
    from RL4CRN.search_streams.julia_stream import JuliaRecipe
    from dataclasses import asdict
    for policy in ('abort','warn'):
        original=asdict(JuliaRecipe(loss_disagreement_policy=policy))
        actual=pilot_julia_recipe({'julia_recipe':original})
        assert actual['loss_disagreement_policy']=='warn'
        assert {k:v for k,v in actual.items() if k!='loss_disagreement_policy'} == {
            k:v for k,v in original.items() if k!='loss_disagreement_policy'}
        assert original['loss_disagreement_policy']==policy


def test_observed_rpa_disagreement_is_recorded_and_returns_canonical_best(tmp_path):
    from collections import defaultdict
    import queue
    import threading
    from RL4CRN.search_streams.julia_stream import JuliaFitPool, JuliaRecipe
    from RL4CRN.search_streams.store import CompactStore
    pool=JuliaFitPool.__new__(JuliaFitPool)
    pool.available=queue.Queue();pool.available.put(object())
    pool.output=tmp_path;pool.recipe=JuliaRecipe(extensions=0,loss_disagreement_policy='warn')
    pool.timeout=None;pool.lock=threading.Lock();pool.extended=defaultdict(int)
    pool.compute_seconds=0.;pool.completed_jobs=0;pool.job_estimate=1.
    pool.search_pool=SimpleNamespace(compute_seconds=1.)
    pool.store=CompactStore(tmp_path/'s.sqlite',100,'record-only-regression',50)
    canonical_loss=1.6102617143003388
    julia_loss=1.4288676282400807
    pool.canonical_pool=SimpleNamespace(evaluate=lambda c:{'loss':canonical_loss,
        'simulation_complete':True,'task_info':{},'candidate_hash':'fit'})
    pool._fit=lambda *args:{'best_rates':[.1],'reaction_ids':[7],'best_loss':julia_loss,
        'evaluations':1,'scenario_solves':9,'traces':[]}
    job={'candidate':{'rates':[1.],'reaction_ids':[7],'loss':5.},
        'seed':1,'epoch':33,'job_key':'recorded-failure-replay','novel':False}
    try:
        result=pool.run(job)
        assert result['termination']=='completed'
        assert result['best']['loss']==canonical_loss
        assert result['loss_agreement']['large_disagreement'] is True
        assert result['loss_agreement']['status']=='warning'
        assert result['loss_agreement']['relative_difference']==pytest.approx(.11264882251707393)
        event=json.loads(pool.store.db.execute("SELECT payload FROM events WHERE kind='julia_loss_agreement'").fetchone()[0])
        assert event['julia_loss']==julia_loss and event['canonical_loss']==canonical_loss
        assert event['policy']['mismatch_action']=='warn'
        saved=json.loads(next(tmp_path.glob('*.json')).read_text())
        assert 'fatal_error' not in saved
        assert saved['outcome']['best']['loss']==canonical_loss
    finally:
        pool.store.close()
