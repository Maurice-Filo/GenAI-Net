import json

import pytest

from comparisons.search_exploration.run_parity12_llm_pilot import (
    CONFIG, audit_workspace, check_evaluator, task_config,
)


def workspace(tmp_path):
    context=tmp_path/'CONTEXT';context.mkdir()
    (context/'HALL_OF_FAME.json').write_text(json.dumps({'entries':[]}))
    (context/'SEARCH_STATE.json').write_text(json.dumps({'hall_of_fame_size':0,
        'llm_feedback':'No previous LLM feedback is available.',
        'llm_best':'No LLM-generated candidate has been accepted yet.'}))
    for name in ('SIL_STATUS.md','EXCLUDED_TOPOLOGIES.md'):
        (context/name).write_text('RL information withheld.')
    return tmp_path


def test_context_allows_own_feedback_only_after_initial_round(tmp_path):
    root=workspace(tmp_path)
    assert audit_workspace(root,0)['hof_entries']==0
    state=root/'CONTEXT/SEARCH_STATE.json';data=json.loads(state.read_text())
    data['llm_feedback']='Prior own proposal: loss=0.1';state.write_text(json.dumps(data))
    audit_workspace(root,1)
    with pytest.raises(ValueError): audit_workspace(root,0)
    with pytest.raises(ValueError): audit_workspace(root,1,True)


def test_context_rejects_hof_and_search_state(tmp_path):
    root=workspace(tmp_path)
    file=root/'CONTEXT/HALL_OF_FAME.json';file.write_text(json.dumps({'entries':[{'loss':.01}]}))
    with pytest.raises(ValueError): audit_workspace(root,1)
    file.write_text(json.dumps({'entries':[]}))
    (root/'CONTEXT/EXPLORATION_STATE.json').write_text('{}')
    with pytest.raises(ValueError): audit_workspace(root,1)


def test_task_copy_has_aligned_bounds_without_modifying_historical_config():
    before=CONFIG.read_bytes();config=task_config()
    assert config['search']['rate_constant_range']==[.001,100.]
    assert config['search']['max_added_reactions']==12
    assert config['logic_parity']['solver']=='CVODE'
    assert CONFIG.read_bytes()==before


def test_real_canonical_contract_and_static_mask():
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.llm.harness_client import build_crn_output_contract
    evaluator,task,_=build('logic_parity',task_config(),.001,100.)
    check_evaluator(evaluator,task)
    contract=build_crn_output_contract(evaluator,num_candidates=10,task_description='Parity',
        candidate_validation_policy='independent-members')
    assert contract['rules']['reaction_count_per_candidate']==12
    assert len(contract['template_reaction_ids'])==4
    assert len(contract['forbidden_reaction_ids'])==5
    assert contract['rules']['allowed_parameter_range']==[.001,100.]
