import hashlib
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from RL4CRN.llm.benchmark_prompts import (
    CRN_AGENT_SYSTEM_PROMPT, get_mmc2_task_prompt, get_mmc2_task_prompt_variant,
)
from RL4CRN.llm.graphs import default_decider_writer_spec
from RL4CRN.llm.harness_client import HarnessLLMClient, build_harness_bot_task
from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
from RL4CRN.llm.schemas import CandidateEvaluation
from RL4CRN.llm import workspace_loss_protocol as protocol
from comparisons.llm_crn_generation import run_parity4_communication_300epoch_queue as queue


def test_legacy_prompt_bytes_are_unchanged():
    texts = {"system": CRN_AGENT_SYSTEM_PROMPT,
             **{t:get_mmc2_task_prompt(t, solver="CVODE") for t in ("logic","rpa","logic_parity")},
             **{k:build_harness_bot_task(k,"calls/001/request.txt") for k in ("text","json")},
             **{n.name:n.prompt_template for n in default_decider_writer_spec().nodes if n.name in ("Decider","Writer")}}
    expected = {
        "system":"56e30312103dc0fc194c7cbd50dddbcde31a716ffab1ba99b0b36a1803e18384",
        "logic":"0395b03800505ce94a0b79c98cdab1b1c2fac288c4f45d4137750288f2d7b3e1",
        "rpa":"4a02b0ef1b28e49f42a16af64cc96dcf6f39772478671f4d6751b2e2f9e32c82",
        "logic_parity":"a4036fa65a342a8bedfdf26408363576e1d1b0da37708a23f35f0a2f6a173298",
        "text":"06b804f7c08279a5707772358d9172740233dd0ea81b076da5cdaf8f1c9336c8",
        "json":"e5352670c76126c784c5e5cd9c280d0626c487f16f0c2cb7d6b97788e9be96e8",
        "Decider":"88aa5c841570b12e30ee8a6b8ce2b3e64a415469f25c40c1a72b5e9ab70eb37d",
        "Writer":"0d752caf610f31c0cef9c17d7dc4e732d6439de999040614135e8267dfa18c2e",
    }
    assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()} == expected


def test_variant_preserves_executable_task_and_has_no_read_quota():
    baseline = get_mmc2_task_prompt("logic_parity", solver="CVODE")
    variant = get_mmc2_task_prompt_variant("logic_parity", variant=protocol.TASK_VARIANT, solver="CVODE")
    assert baseline.split("Use the received Hall of Fame actively.")[0] == variant.split("Evidence-guided Decider policy")[0]
    assert "six evidence-guided refinements" in variant
    for text in (protocol.SYSTEM_PROMPT, protocol.bot_task("text","request.md"), protocol.bot_task("json","request.md")):
        assert "eight workspace-tool" not in text
        assert "no fixed workspace-tool-call quota" in text.lower()
    with pytest.raises(ValueError):
        get_mmc2_task_prompt_variant("logic", variant=protocol.TASK_VARIANT)


def test_four_arm_replacement_has_new_ids_and_keeps_budget_and_information_controls():
    for mode in queue.MODES:
        command = queue.campaign_command(mode, protocol.PROTOCOL)
        assert queue.campaign_id(mode) != queue.campaign_id(mode, protocol.PROTOCOL)
        assert queue.audit_path(mode) != queue.audit_path(mode, protocol.PROTOCOL)
        assert command[command.index("--task-prompt-variant")+1] == protocol.TASK_VARIANT
        assert command[command.index("--communication-mode")+1] == mode
        assert command[command.index("--epochs")+1] == "300"
        assert command[command.index("--total-candidate-budget")+1] == "307200"
        assert command[command.index("--max-agent-evaluations")+1] == "0"
        assert "--withhold-initial-hof" in command
        assert "--comet-project" in command
        assert "--task-prompt-variant" in queue.audit_command(mode, protocol.PROTOCOL)
    assert queue.schedule_path() != queue.schedule_path(protocol.PROTOCOL)


def test_real_parity_preflight_uses_production_llm_bounds_and_template_mask():
    definition = queue.validate_task_definition(protocol.PROTOCOL)
    assert definition["production_llm_rate_bounds"] == [.001,100.]
    assert definition["masked_reaction_ids"] == [0,1,2,3,4]
    assert definition["reaction_budget"] == 8
    assert "[0.001, 100.0]" in definition["design_constraints_example"]


def test_replacement_rejects_incompatible_settings_before_launch():
    args = SimpleNamespace(task_prompt_variant=protocol.TASK_VARIANT,tasks=["logic_parity"],
        generation_backend="harness",candidate_validation_policy="independent-members",
        max_agent_evaluations=0,max_literature_searches=0,literature_rag_index=None,withhold_initial_hof=True)
    protocol.validate_campaign_settings(args)
    for key,value in (("max_agent_evaluations",1),("withhold_initial_hof",False),
                      ("tasks",["logic"]),("generation_backend","direct-nonthinking")):
        with pytest.raises(ValueError):
            protocol.validate_campaign_settings(SimpleNamespace(**{**vars(args),key:value}))


class Library:
    def __len__(self):
        return 3

    def get_reaction(self, index):
        return SimpleNamespace(num_parameters=1, ID=index)

    def find_zero_reaction(self):
        return 0


class Evaluator:
    library = Library()
    max_added_reactions = 1
    require_unique_reactions = True
    initial_reaction_ids = frozenset({0})
    forbidden_reaction_ids = frozenset({0})
    enforce_parameter_bounds = True
    min_parameter_value = .001
    max_parameter_value = 100.

    def evaluate_many(self, candidates, **kwargs):
        return [CandidateEvaluation(candidate=c,valid=True,loss=.5) for c in candidates]


def test_decider_writer_end_to_end_without_exploratory_service(tmp_path):
    client = HarnessLLMClient(workspace_root=tmp_path/"harness-workspaces",dsh_home=tmp_path/"home",
        command=["fake-dsh"],candidate_validation_policy="independent-members",harness_protocol=protocol.PROTOCOL)
    clone = client.fork()
    assert clone.harness_protocol == protocol.PROTOCOL
    assert clone.system_prompt == client.system_prompt
    graph = HarnessDeciderWriterCRNGraph(client=client,evaluator=Evaluator(),writer_retry_limit=0,
        max_workspace_evaluations=0,max_literature_searches=0)
    decision = "1: reaction 1 at rate 200. 2: unencodable design. 3: reaction 2 at rate 0.00001."
    payload = {"candidates":[{"reaction_ids":[1],"parameter_values":[[200]]},
                              {"reaction_ids":[],"parameter_values":[]},
                              {"reaction_ids":[2],"parameter_values":[[.00001]]}]}
    replies = [subprocess.CompletedProcess([],0,stdout=decision,stderr=""),
               subprocess.CompletedProcess([],0,stdout=json.dumps(payload),stderr="")]
    with patch.object(client,"_initialize_git"), patch(
        "RL4CRN.llm.harness_client.subprocess.run",side_effect=replies) as run, patch(
        "RL4CRN.llm.harness_runner.WorkspaceEvaluationService") as service:
        result = graph.run_round(task_description=protocol.parity_task_prompt(solver="CVODE"),num_candidates=3)
    assert run.call_count == 2
    service.assert_not_called()
    assert len(result.evaluations) == 2
    assert result.response_validation["accepted_candidate_indices"] == [0,2]
    assert result.response_validation["rejected_candidates"][0]["candidate_index"] == 1
    assert result.raw_payload["candidates"][0]["parameter_values"] == [[100.]]
    assert result.raw_payload["candidates"][1]["parameter_values"] == [[.001]]
    root = client.last_workspace.path
    constraints = (root/"DESIGN_CONSTRAINTS.md").read_text()
    assert "Forbidden added reaction IDs: [0]" in constraints
    assert "[0.001, 100.0]" in constraints
    assert "REACTION_LIBRARY.tsv" in constraints
    assert not (root/".dsh/skills/crn-simulation/SKILL.md").exists()
    assert not (root/"CONTEXT/evaluator_endpoint.json").exists()
    assert json.loads((root/"CONTEXT/HALL_OF_FAME.json").read_text())["entries"] == []
    guide = json.loads((root/"OUTPUT_GUIDE.json").read_text())
    assert guide["unencodable_member"]["scientific_repairs_allowed"] is False
    for call in run.call_args_list:
        assert "eight workspace-tool" not in call.args[0][-1]
        assert "DESIGN_CONSTRAINTS.md" in call.args[0][-1]
    assert not protocol.audit_workspaces(root.parent.parent)


def test_protocol_refuses_tools_and_atomic_batch(tmp_path):
    with pytest.raises(ValueError):
        HarnessLLMClient(workspace_root=tmp_path,dsh_home=tmp_path,harness_protocol=protocol.PROTOCOL)
    client = HarnessLLMClient(workspace_root=tmp_path,dsh_home=tmp_path,
        harness_protocol=protocol.PROTOCOL,candidate_validation_policy="independent-members")
    with pytest.raises(ValueError):
        HarnessDeciderWriterCRNGraph(client=client,evaluator=Evaluator(),max_workspace_evaluations=1)


def test_new_protocol_release_is_hash_pinned_and_does_not_replace_legacy_approval(tmp_path):
    review_path, approval_path = tmp_path/"review.json", tmp_path/"approval.json"
    assert protocol.validate_protocol_release("standard",review_path=review_path,approval_path=approval_path) == {}
    with pytest.raises(RuntimeError):
        protocol.validate_protocol_release(protocol.TASK_VARIANT,review_path=review_path,approval_path=approval_path)
    review_path.write_text(json.dumps(protocol.review_payload()))
    approval_path.write_text(json.dumps({"approval_status":"approved","approved_by":"test fixture",
        "approved_at":"test fixture","prompt_review_sha256":hashlib.sha256(review_path.read_bytes()).hexdigest()}))
    protocol.validate_protocol_release(protocol.TASK_VARIANT,review_path=review_path,approval_path=approval_path)
    review_path.write_text(review_path.read_text()+"\n")
    with pytest.raises(RuntimeError):
        protocol.validate_protocol_release(protocol.TASK_VARIANT,review_path=review_path,approval_path=approval_path)


def test_runtime_approval_matches_frozen_replacement_and_viewer_points_to_new_ids():
    from comparisons.llm_crn_generation import run_parity12_communication_queue as current_queue

    protocol.validate_protocol_release(protocol.TASK_VARIANT)
    plan = json.loads((protocol.GENERATED/"contract_v2_flash_logic_recovery_progress.json").read_text())
    parity = [r for r in plan["campaigns"] if r["tasks"] == ["logic_parity"]]
    assert {r["id"] for r in parity} == {current_queue.campaign_id(mode) for mode in queue.MODES}
    assert all(r["phase"] in {"planned", "queued", "running", "completed", "attention"} for r in parity)
    assert all(r["seeds"] == 20 and r["epochs"] == 300 for r in parity)
