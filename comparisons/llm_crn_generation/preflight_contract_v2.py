#!/usr/bin/env python3
"""Build and audit all deterministic contract-v2 proposal spaces without model calls."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper
from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
from RL4CRN.llm.benchmark_prompts import get_mmc2_task_prompt_variant
from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
from RL4CRN.llm.harness_client import build_crn_output_contract
from RL4CRN.llm.schemas import LLMCandidate
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import BUILDERS, CONFIGS


TASKS = (
    "rpa",
    "logic",
    "classifier",
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "oscillator_mean",
    "oscillator_frequency",
)
OUTPUT = ROOT / "paper/iclr2027_genai_net_llm/generated/contract_v2_preflight"
RATE_MIN = 0.001
RATE_MAX = 100.0


def digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def negative_candidate(evaluator: LLMCandidateEvaluator, forbidden_id: int) -> LLMCandidate:
    allowed = [
        reaction_id
        for reaction_id in range(len(evaluator.library))
        if reaction_id not in evaluator.forbidden_reaction_ids
    ]
    reaction_ids = [forbidden_id, *allowed[: evaluator.max_added_reactions - 1]]
    parameter_values = [
        [1.0] * int(evaluator.library.get_reaction(reaction_id).num_parameters)
        for reaction_id in reaction_ids
    ]
    return LLMCandidate(reaction_ids, parameter_values, "preflight negative test")


def audit_task(task_name: str) -> dict:
    config = json.loads(CONFIGS[task_name].read_text(encoding="utf-8"))
    config[task_name]["solver"] = "CVODE"
    crn, library_components, task, _cfg = BUILDERS[task_name](config)
    library = library_components[0]
    null_id = int(library.find_zero_reaction())
    template_ids = sorted(int(value) for value in crn.gather_reaction_IDs())
    evaluator = LLMCandidateEvaluator(
        crn_template=crn,
        max_added_reactions=int(config["search"]["max_added_reactions"]),
        library=library,
        stepper=IOCRNStepper(),
        actuator=LibraryActuator(library),
        compute_reward_func=task.compute_reward,
        min_parameter_value=RATE_MIN,
        max_parameter_value=RATE_MAX,
        enforce_parameter_bounds=True,
        forbidden_reaction_ids=[null_id],
    )
    prompt = get_mmc2_task_prompt_variant(
        task_name, variant="standard", solver="CVODE"
    )
    contract = build_crn_output_contract(
        evaluator,
        num_candidates=10,
        task_description=prompt,
        candidate_validation_policy="independent-members",
    )
    contract_ids = {int(item["id"]) for item in contract["reaction_library"]}
    schema_ids = set(
        contract["json_schema"]["properties"]["candidates"]["items"]["properties"]
        ["reaction_ids"]["items"]["enum"]
    )
    forbidden = set(template_ids) | {null_id}
    rl_signature = np.asarray(crn.get_bool_signature(), dtype=bool)
    failures = []
    if contract["contract_version"] != 2:
        failures.append("contract version is not 2")
    if contract["template_reaction_ids"] != template_ids:
        failures.append("template IDs are not recorded exactly")
    if contract["null_reaction_id"] != null_id:
        failures.append("null ID is not recorded exactly")
    if forbidden & contract_ids or forbidden & schema_ids:
        failures.append("fixed or null ID appears in the model-facing proposal space")
    if any(not bool(rl_signature[reaction_id]) for reaction_id in template_ids):
        failures.append("a template ID is not present in the initial RL state mask")
    if contract["rules"]["allowed_parameter_range"] != [RATE_MIN, RATE_MAX]:
        failures.append("rate bounds disagree")
    if contract["rules"]["out_of_range_parameter_policy"] != "clamp":
        failures.append("rate clamp policy is absent")
    if contract["rules"]["candidate_validation_policy"] != "independent-members":
        failures.append("member-validation policy disagrees")
    if "[0.001, 100]" not in prompt:
        failures.append("task prompt does not state contract-v2 rate bounds")
    target_forbidden = template_ids[0] if template_ids else null_id
    negative = evaluator.evaluate(negative_candidate(evaluator, target_forbidden))
    if negative.valid or "forbidden" not in negative.message.lower():
        failures.append("deliberate forbidden-ID candidate was not rejected pre-simulation")

    result = {
        "task": task_name,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "proposal_space_contract_version": 2,
        "template_reaction_ids": template_ids,
        "null_reaction_id": null_id,
        "forbidden_reaction_ids": sorted(evaluator.forbidden_reaction_ids),
        "allowed_reaction_count": len(contract_ids),
        "allowed_reaction_ids_sha256": digest_json(sorted(contract_ids)),
        "output_contract_sha256": digest_json(contract),
        "task_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rate_bounds": [RATE_MIN, RATE_MAX],
        "candidate_validation_policy": "independent-members",
        "negative_test": {
            "reaction_id": target_forbidden,
            "valid": bool(negative.valid),
            "message": negative.message,
        },
    }
    (OUTPUT / f"{task_name}_output_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [audit_task(task) for task in TASKS]
    report = {
        "status": "pass" if all(row["status"] == "pass" for row in results) else "fail",
        "model_calls_made": 0,
        "author_prompt_approval": "pending",
        "tasks": results,
    }
    (OUTPUT / "preflight_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Contract-v2 static preflight",
        "",
        f"Overall status: **{report['status'].upper()}**",
        "",
        "No model calls were made. Prompt approval and the dynamic sentinel remain blocking gates.",
        "",
        "| Task | Fixed IDs | Null ID | Allowed | Negative test | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['task']} | {row['template_reaction_ids']} | {row['null_reaction_id']} | "
            f"{row['allowed_reaction_count']} | {row['negative_test']['message']} | "
            f"{row['status']} |"
        )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "pass":
        raise SystemExit("Contract-v2 static preflight failed; inspect preflight_report.json")
    print(OUTPUT / "preflight_report.json")


if __name__ == "__main__":
    main()
