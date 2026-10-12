import json

from comparisons.search_exploration.check_rpa_flash_harness import PROMPT, prepare
from RL4CRN.llm.harness_client import build_crn_output_contract
from RL4CRN.llm.harness_runner import _workspace_context_files
from RL4CRN.llm.workspace_loss_protocol import PROTOCOL, SYSTEM_PROMPT, graph_spec


def test_readiness_preserves_new_roles_and_rpa_constraints(tmp_path):
    config, client, graph = prepare(tmp_path, tmp_path / "dsh-home")
    assert client.model == "deepseek-v4-flash"
    assert client.harness_protocol == PROTOCOL
    assert client.system_prompt == SYSTEM_PROMPT.strip()
    assert graph.spec == graph_spec()
    assert graph.writer_retry_limit == 0
    assert graph.max_workspace_evaluations == graph.max_literature_searches == 0
    assert graph.evaluator.forbidden_reaction_ids == frozenset({0, 1, 28})
    assert config["search"]["rate_constant_range"] == [.001, 100.]
    assert "six evidence-guided refinements" in PROMPT.read_text()
    contract = build_crn_output_contract(
        graph.evaluator, num_candidates=10, task_description=PROMPT.read_text(),
        candidate_validation_policy=client.candidate_validation_policy,
    )
    assert contract["rules"]["allowed_parameter_range"] == [.001, 100.]
    assert contract["rules"]["out_of_range_parameter_policy"] == "clamp"


def test_readiness_workspace_has_no_initial_hof_or_simulation_tools(tmp_path):
    _, client, graph = prepare(tmp_path, tmp_path / "dsh-home")
    contract = build_crn_output_contract(
        graph.evaluator, num_candidates=10, task_description=PROMPT.read_text(),
        candidate_validation_policy=client.candidate_validation_policy,
    )
    files = _workspace_context_files({"step": 0, "hall_of_fame_iter": None}, graph.memory)
    with client.run(task_description=PROMPT.read_text(), contract=contract,
                    workspace_files=files, label="offline-test") as workspace:
        assert workspace.call_count == 0
        root = workspace.path
        assert json.loads((root / "CONTEXT/HALL_OF_FAME.json").read_text())["entries"] == []
        assert not (root / "CONTEXT/hall-of-fame").exists()
        assert not (root / "CONTEXT/evaluator_endpoint.json").exists()
        assert not (root / ".dsh/skills/crn-simulation/SKILL.md").exists()
        assert "Forbidden added reaction IDs: [0, 1, 28]" in (root / "DESIGN_CONSTRAINTS.md").read_text()
        assert json.loads((root / "run_manifest.json").read_text())["exploratory_evaluations_enabled"] is False
