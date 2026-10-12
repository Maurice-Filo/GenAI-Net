"""Isolated Flash/Harness parity proposal pilot; no RL, fitting or outward state."""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT/'comparisons/search_exploration/prompts/parity12-evidence-exploitation-v2.md'
CONFIG = ROOT/'comparisons/rpa_search/configs/logic_parity_12reactions_300epoch.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_config():
    config = read(CONFIG)
    config['search']['rate_constant_range'] = [.001,100.]
    return config


def check_evaluator(evaluator, task):
    if evaluator.max_added_reactions != 12 or len(evaluator.initial_reaction_ids) != 4:
        raise ValueError('Unexpected parity template/budget')
    if evaluator.forbidden_reaction_ids != evaluator.initial_reaction_ids | {evaluator.library.find_zero_reaction()}:
        raise ValueError('Unexpected parity mask')
    if task.kind != 'logic' or task.norm != 1 or str(evaluator.crn_template.solver).upper() != 'CVODE':
        raise ValueError('Wrong canonical evaluator')
    if (evaluator.min_parameter_value,evaluator.max_parameter_value) != (.001,100.):
        raise ValueError('Wrong rate bounds')


def audit_workspace(path, round_index, independent=False):
    hof = read(path/'CONTEXT/HALL_OF_FAME.json')
    state = read(path/'CONTEXT/SEARCH_STATE.json')
    if hof['entries'] or state['hall_of_fame_size'] or (path/'CONTEXT/hall-of-fame').exists():
        raise ValueError('RL/HOF state exposed to no-communication pilot')
    if (path/'CONTEXT/EXPLORATION_STATE.json').exists() or (path/'CONTEXT/evaluator_endpoint.json').exists():
        raise ValueError('Search state or exploratory evaluator exposed')
    for name in ('SIL_STATUS.md','EXCLUDED_TOPOLOGIES.md'):
        if 'withheld' not in (path/'CONTEXT'/name).read_text().lower():
            raise ValueError('Missing explicit outward-state withholding')
    if round_index == 0 or independent:
        if state['llm_feedback'] != 'No previous LLM feedback is available.' or state['llm_best'] != 'No LLM-generated candidate has been accepted yet.':
            raise ValueError('Initial/independent request has prior memory')
    return {'hof_entries':0,'outward_state':'withheld','self_feedback':state['llm_feedback'],
            'self_best':state['llm_best']}


def plot_progress(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    valid = [r for r in rows if r.get('best_loss') is not None]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    if valid:
        axes[0].step([r['round']+1 for r in valid],[r['best_loss'] for r in valid],where='post',color='#a64e85')
        axes[0].set_yscale('log')
    axes[0].set(xlabel='Completed request rounds',ylabel='Best complete canonical loss')
    axes[1].bar([r['round']+1 for r in rows],[r.get('complete_candidates',0) for r in rows],color='#a64e85')
    axes[1].set(xlabel='Completed request rounds',ylabel='Complete scored proposals',ylim=(0,10))
    fig.savefig(output/'progress.png',dpi=160);fig.savefig(output/'progress.pdf');plt.close(fig)


def run(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JULIA_NUM_THREADS'):
        os.environ[key]='1'
    os.environ.setdefault('MPLCONFIGDIR','/local0/tmp/mpl')
    from RL4CRN.search_exploration.engine import atomic_json
    from RL4CRN.search_exploration.evaluation import build, EvaluatorPool, EvaluationService
    from RL4CRN.search_exploration.store import fingerprint
    from RL4CRN.search_streams.store import CompactStore
    from RL4CRN.search_streams.harness import StreamHarnessEvaluator
    from RL4CRN.llm.harness_client import HarnessLLMClient, HarnessResponseError, build_crn_output_contract
    from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
    from RL4CRN.llm.memory import LLMMemory, LLMFeedbackRecord
    from RL4CRN.llm.workspace_loss_protocol import PROTOCOL, SYSTEM_PROMPT, graph_spec
    output=args.output.resolve()
    if not args.live:
        output.mkdir(parents=True,exist_ok=False)
        config=task_config()
        evaluator,task,_=build('logic_parity',config,.001,100.)
        check_evaluator(evaluator,task)
        contract=build_crn_output_contract(evaluator,num_candidates=10,task_description=PROMPT.read_text(),
            candidate_validation_policy='independent-members')
        atomic_json(output/'task.json',config);atomic_json(output/'output_contract.json',contract)
        shutil.copy2(PROMPT,output/'TASK.md')
        frozen=[Path(__file__).resolve(),PROMPT,CONFIG]
        for folder in ('RL4CRN/llm','RL4CRN/search_streams','RL4CRN/search_exploration'):
            frozen.extend(sorted((ROOT/folder).glob('*.py')))
        atomic_json(output/'manifest.json',{'kind':'parity12-llm-only-pilot','rounds':args.rounds,
            'candidates_per_round':10,'mode':args.mode,'harness_protocol':PROTOCOL,
            'model':'deepseek-v4-flash','provider':'deepseek-official','rate_bounds':[.001,100.],
            'task_prompt_sha256':sha(output/'TASK.md'),'task_config_sha256':sha(output/'task.json'),
            'system_prompt_sha256':hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            'role_prompts':{n.name:n.prompt_template for n in graph_spec().nodes if n.name in ('Decider','Writer')},
            'no_rl':True,'no_rate_fitting':True,'no_rag':True,'exploratory_evaluations':0,
            'withhold_hof_all_rounds':True,'validation':'independent-members','writer_retries':0,
            'llm_role_timeout_seconds':900,'canonical_evaluation_timeout_seconds':None,
            'comet_project':args.comet_project,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in frozen}})
        atomic_json(output/'status.json',{'status':'prepared','completed_rounds':0,'planned_rounds':args.rounds})
        print(json.dumps({'status':'prepared','output':str(output),'forbidden_ids':sorted(evaluator.forbidden_reaction_ids)}),flush=True)
        return
    manifest=read(output/'manifest.json')
    if manifest['mode']!=args.mode or manifest['rounds']!=args.rounds or manifest['comet_project']!=args.comet_project:
        raise ValueError('Launch settings differ from preparation')
    for file,h in manifest['source_hashes'].items():
        if sha(ROOT/file)!=h: raise ValueError(f'Frozen source changed: {file}')
    if sha(output/'TASK.md')!=manifest['task_prompt_sha256'] or sha(output/'task.json')!=manifest['task_config_sha256']:
        raise ValueError('Frozen prompt/task changed')
    lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(output/'status.json')['status']!='prepared':
        raise ValueError('Pilot already started; no blind retry/resume')
    started=time.monotonic();logger=pool=store=None
    history=[];best=None;topologies={};rows_correct=0
    try:
        from comparisons.search_exploration.recover_rpa_directional_balance import balance_status
        balance=balance_status();atomic_json(output/'balance_initial.json',balance)
        if not balance['is_available']: raise RuntimeError('Provider balance unavailable')
        os.environ.pop('COMET_API_KEY',None)
        from comet_ml import Experiment
        logger=Experiment(project_name=args.comet_project,auto_param_logging=False,auto_metric_logging=False,
            log_code=False,log_git_patch=False,log_graph=False,log_git_metadata=False,
            log_env_details=False,log_env_host=False,log_env_cpu=False,log_env_gpu=False,display_summary_level=0)
        if not logger.get_key() or not getattr(logger,'alive',False): raise RuntimeError('Comet authentication failed')
        logger.set_name(output.name);logger.add_tags(['parity12','flash','llm-only','no-communication','prompt-v2'])
        logger.log_parameters({k:v for k,v in manifest.items() if k not in ('source_hashes','role_prompts')})
        atomic_json(output/'comet.json',{'project':args.comet_project,'experiment_key':logger.get_key()})
        config=read(output/'task.json');base,task,_=build('logic_parity',config,.001,100.)
        check_evaluator(base,task)
        store=CompactStore(output/'exploration.sqlite',args.rounds*10,fingerprint(manifest),0,wait_timeout=None)
        pool=EvaluatorPool(('logic_parity',config,.001,100.),workers=1,timeout=None)
        allowed=[r.ID for r in base.library.reactions if r.ID not in base.forbidden_reaction_ids]
        service=EvaluationService(store,pool,allowed,12,(.001,100.))
        evaluator=StreamHarnessEvaluator(base,service)
        client=HarnessLLMClient(workspace_root=output/'harness-workspaces',dsh_home=args.dsh_home,
            provider='deepseek-official',model='deepseek-v4-flash',timeout_seconds=900,
            candidate_validation_policy='independent-members',harness_protocol=PROTOCOL)
        graph=HarnessDeciderWriterCRNGraph(client=client,evaluator=evaluator,writer_retry_limit=0,
            max_workspace_evaluations=0,max_literature_searches=0,comet_logger=logger)
        for index in range(args.rounds):
            if args.mode=='independent': graph.memory=LLMMemory()
            atomic_json(output/'status.json',{'status':'running','active_round':index,'completed_rounds':len(history),
                'planned_rounds':args.rounds,'best_loss':best,'elapsed_seconds':time.monotonic()-started})
            tic=time.monotonic();row={'round':index};evaluations=[]
            try:
                result=graph.run_round(task_description=(output/'TASK.md').read_text(),num_candidates=10,step=index,
                    hall_of_fame_iter=None,forbidden_topologies_text='RL-derived topology exclusions withheld in every round. Static reaction masks still apply.',
                    sil_feedback_text='RL/HOF/SIL state withheld in every round. Only LLM self-feedback is permitted.',
                    jsonl_path=output/'evaluations.jsonl')
                evaluations=result.evaluations
                row.update(status='completed',validation=result.response_validation,
                    structurally_accepted=len(evaluations),complete_candidates=sum(e.valid for e in evaluations))
            except HarnessResponseError as error:
                row.update(status='response_failed',error_type=type(error).__name__,error=str(error),
                    structurally_accepted=0,complete_candidates=0)
                graph.memory.feedback.append(LLMFeedbackRecord('',False,None,'Previous round failed response validation; no replacement batch was generated.'))
            if evaluator.fatal_error is not None: raise RuntimeError('Canonical evaluator infrastructure failed') from evaluator.fatal_error
            workspace=client.last_workspace.path
            audit=audit_workspace(workspace,index,args.mode=='independent')
            atomic_json(workspace/'no_communication_audit.json',audit)
            records=[]
            for e in evaluations:
                r=getattr(e,'exploration_record',None)
                if r:
                    clean={k:v for k,v in r.items() if k!='task_info'}
                    records.append({'valid':e.valid,'message':e.message,'record':clean})
                    if e.valid:
                        best=r['loss'] if best is None else min(best,r['loss'])
                        h=r['topology_hash'];topologies[h]=min(topologies.get(h,float('inf')),r['loss'])
                        rows_correct+=int(bool(r.get('all_rows_correct')))
            atomic_json(workspace/'canonical_records.json',records)
            row.update(workspace=str(workspace),role_invocations=client.last_workspace.call_count,
                elapsed_seconds=time.monotonic()-tic,best_loss=best,complete_topologies=len(topologies),
                all_rows_correct_proposals=rows_correct)
            history.append(row);atomic_json(output/'history.json',history)
            atomic_json(output/'self_memory.json',{'feedback':[asdict(r) for r in graph.memory.feedback],
                'best_text':graph.memory.format_best(),'rounds':len(history)})
            plot_progress(output,history)
            logger.log_metrics({k:row[k] for k in ('complete_candidates','complete_topologies','role_invocations','all_rows_correct_proposals')},step=index)
            if best is not None: logger.log_metric('best_canonical_loss',best,step=index)
            logger.log_image(str(output/'progress.png'),name='proposal_progress',step=index)
            print(json.dumps(row),flush=True)
            atomic_json(output/'status.json',{'status':'between_rounds','completed_rounds':len(history),
                'planned_rounds':args.rounds,'best_loss':best,'complete_topologies':len(topologies)})
            if (index+1)%5 == 0:
                try:
                    sample=balance_status()
                    atomic_json(output/f'balance_after_round_{index+1:02d}.json',sample)
                    for b in sample['balances']:
                        if b['currency']=='USD': logger.log_metric('account_balance_usd',float(b['total_balance']),step=index)
                    if not sample['is_available']: raise RuntimeError('Provider balance exhausted')
                except RuntimeError:
                    raise
                except Exception as error:
                    atomic_json(output/f'balance_check_failure_{index+1:02d}.json',{'type':type(error).__name__})
        final={'status':'complete','completed_rounds':len(history),'planned_rounds':args.rounds,
            'best_loss':best,'complete_topologies':len(topologies),'all_rows_correct_proposals':rows_correct,
            'failed_response_rounds':sum(r['status']=='response_failed' for r in history),
            'counters':store.counters(),'elapsed_seconds':time.monotonic()-started}
        atomic_json(output/'completed.json',final);atomic_json(output/'status.json',final)
        try: atomic_json(output/'balance_final.json',balance_status())
        except Exception as error: atomic_json(output/'balance_check_failure.json',{'type':type(error).__name__})
        logger.log_other('status','complete')
    except BaseException as error:
        failure={'status':'failed','type':type(error).__name__,'message':str(error),'completed_rounds':len(history)}
        atomic_json(output/'failed.json',failure);atomic_json(output/'status.json',failure)
        if logger: logger.log_other('status','failed')
        raise
    finally:
        if pool: pool.close()
        if store: store.close()
        if logger: logger.end()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=20)
    parser.add_argument('--mode',choices=['sequential','independent'],default='sequential')
    parser.add_argument('--comet-project',default='genai-net-exploration')
    parser.add_argument('--dsh-home',type=Path,default=Path.home()/'ai-workspaces/deepseek-test/.dsh-home')
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if not 1<=args.rounds<=20: parser.error('Pilot permits 1..20 rounds')
    run(args)
