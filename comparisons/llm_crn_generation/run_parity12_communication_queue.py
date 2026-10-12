"""Isolated 12-added-reaction replacement; never resume or overwrite parity-8."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation import run_parity4_communication_300epoch_queue as old
from RL4CRN.llm import workspace_loss_protocol as protocol
from RL4CRN.llm.harness_client import build_crn_output_contract

VARIANT = protocol.TWELVE_VARIANT
SCHEDULE = old.GENERATED / "parity12_communication_300epoch_20260909_schedule.json"
LOGS = old.GENERATED / "parity12_20260909_logs"


def campaign_id(mode):
    return f"flash-parity4-12r-workspace-loss-{mode}-300epoch-seeds0to19-20260909"


def method_name(mode):
    return f"genai_net_llm_flash_parity4_12r_workspace_loss_{mode.replace('-', '_')}"


def run_suffix(mode):
    return f"cvode_llm_flash_parity4_12r_workspace_loss_{mode.replace('-', '_')}"


def command(mode, audit=False):
    cmd = (old.audit_command if audit else old.campaign_command)(mode, protocol.PROTOCOL)
    values = {"--method-name": method_name(mode), "--run-suffix": run_suffix(mode),
              "--task-prompt-variant": VARIANT}
    if audit:
        cmd[2] = str(old.CAMPAIGN_ROOT / campaign_id(mode))
        values["--output"] = str(old.GENERATED / f"parity12_{mode.replace('-', '_')}_20260909_audit.json")
    else:
        values["--campaign-id"] = campaign_id(mode)
    for flag, value in values.items():
        cmd[cmd.index(flag)+1] = value
    return cmd


def preflight():
    original = json.loads(old.CONFIGS[old.TASK].read_text())
    path = protocol.config_path_for_variant(old.TASK, VARIANT)
    config = json.loads(path.read_text())
    expected = json.loads(json.dumps(original))
    expected["search"]["max_added_reactions"] = 12
    expected["benchmark"]["name"] = "logic_parity_4_12reactions_300epoch"
    if config != expected:
        raise RuntimeError("Replacement config changes more than reaction capacity and its name")
    evaluator = old.build_evaluator(old.TASK, solver="CVODE", config_path=path)
    evaluator.min_parameter_value, evaluator.max_parameter_value = .001, 100.0
    prompt = protocol.parity_task_prompt(solver="CVODE", variant=VARIANT)
    contract = build_crn_output_contract(evaluator, num_candidates=10, task_description=prompt)
    if evaluator.max_added_reactions != 12 or contract["rules"]["reaction_count_per_candidate"] != 12:
        raise RuntimeError("Evaluator and Writer contract must require twelve reactions")
    allowed = {r["id"] for r in contract["reaction_library"]}
    if allowed.intersection(contract["forbidden_reaction_ids"]):
        raise RuntimeError("Masked reactions are present in the proposal library")
    if "exactly twelve distinct reactions" not in prompt:
        raise RuntimeError("Task prompt does not require twelve reactions")
    review = protocol.review_payload(VARIANT)
    old_review = protocol.review_payload()
    expected_texts = dict(old_review["texts"])
    expected_texts["TASK.md"] = expected_texts["TASK.md"].replace(
        "choose exactly eight distinct reactions", "choose exactly twelve distinct reactions", 1)
    if review["texts"] != expected_texts:
        raise RuntimeError("Replacement changes more than the task reaction count")
    return {"config_path": str(path), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "added_reactions": 12, "fixed_reactions": 4, "total_reactions": 16,
            "allowed_reactions": len(allowed), "masked_ids": contract["forbidden_reaction_ids"],
            "rate_bounds": contract["rules"]["allowed_parameter_range"],
            "task_prompt_sha256": review["sha256"]["TASK.md"]}


def freeze_review():
    payload = protocol.review_payload(VARIANT)
    path, _ = protocol.review_paths(VARIANT)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Refusing to replace a different frozen review")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    lines = ["# Parity, twelve added reactions", "", "Only the reaction count differs from the eight-reaction workspace-loss-v1 protocol."]
    for name, text in payload["texts"].items():
        lines += ["", f"## {name}", "", f"SHA-256: `{payload['sha256'][name]}`", "", "```text", text.rstrip(), "```"]
    (old.GENERATED / "PARITY12_WORKSPACE_LOSS_PROMPTS.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if any((old.CAMPAIGN_ROOT / campaign_id(m)).exists() for m in old.MODES):
        raise SystemExit("Replacement outputs already exist; refusing overwrite/relaunch")
    definition = preflight()
    freeze_review()
    record = {"status": "planned", "task_definition": definition,
              "task_prompt_variant": VARIANT, "model": old.MODEL,
              "replaces": str(old.schedule_path(protocol.PROTOCOL)),
              "campaigns": {m: {"campaign_id": campaign_id(m), "command": command(m),
                                 "audit_command": command(m, audit=True)} for m in old.MODES}}
    def save():
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        SCHEDULE.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    save()
    if args.plan_only:
        print(json.dumps(definition, indent=2))
        return
    protocol.validate_protocol_release(VARIANT)
    LOGS.mkdir(exist_ok=True)
    active = {}
    states = record["arm_status"] = {}
    for mode in old.MODES:
        handle = (LOGS / f"{campaign_id(mode)}.log").open("ab")
        process = subprocess.Popen(command(mode), cwd=ROOT, env=old.environment(), stdout=handle, stderr=subprocess.STDOUT)
        active[mode] = process, handle
        states[mode] = "running"
        print(f"[start] {mode} pid={process.pid}", flush=True)
    record["status"] = "running"
    save()
    while active:
        for mode, (process, handle) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            states[mode] = "complete" if rc == 0 else f"failed:{rc}"
            del active[mode]
            save()
        if active:
            time.sleep(10)
    if any(s != "complete" for s in states.values()):
        record["status"] = "failed"
        save()
        raise SystemExit("One or more parity12 arms failed")
    audits = {m: subprocess.run(command(m, audit=True), cwd=ROOT, env=old.environment()).returncode for m in old.MODES}
    record.update(audit_returncodes=audits, status="complete" if not any(audits.values()) else "audit-failed")
    save()
    if any(audits.values()):
        raise SystemExit("Parity12 audit failed")


if __name__ == "__main__":
    main()
