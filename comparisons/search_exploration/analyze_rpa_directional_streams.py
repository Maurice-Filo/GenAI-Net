"""Read-only endpoint analysis and explicitly post-hoc independent unions.

No model calls, new simulations, fitting, or mutation of source databases.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from RL4CRN.search_streams.directional_harness import topology_union
from RL4CRN.search_exploration.engine import atomic_json

CONTROL_ROOT = Path("/local1/rossin/search-exploration/rpa-statistical-20seed-20260915")


def read(path):
    return json.loads(path.read_text())


def designs(root):
    if read(root / "completed.json")["status"] != "complete":
        raise ValueError("Only completed endpoints may be pooled")
    path = (root / "exploration.sqlite").resolve()
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Database has uncheckpointed writes; retry after clean shutdown")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        return [json.loads(r[0]) for r in db.execute("SELECT payload FROM designs")]


def audit_self_feedback_bank(root):
    manifest = read(root / "manifest.json")
    if manifest["communication_mode"] != "llm-to-rl" or not manifest["withhold_initial_hof"]:
        raise ValueError("Union requires an insertion-only, self-feedback LLM bank")
    workspaces = list((root / "harness-workspaces").glob("*/run_manifest.json"))
    if not workspaces:
        raise ValueError("No saved request contexts to verify")
    for file in workspaces:
        w = file.parent
        if read(w / "CONTEXT/HALL_OF_FAME.json")["entries"]:
            raise ValueError("Search HOF leaked into an insertion-only request")
        if (w / "CONTEXT/EXPLORATION_STATE.json").exists() or (w / "CONTEXT/hall-of-fame").exists():
            raise ValueError("Search diagnostics leaked into insertion-only context")
        for name in ("SIL_STATUS.md", "EXCLUDED_TOPOLOGIES.md"):
            if "withheld" not in (w / "CONTEXT" / name).read_text().lower():
                raise ValueError("Expected explicit withholding of search-derived state")
    return len(workspaces)


def direct_records(root):
    path = root / "direct_llm_proposals.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(not row["direct_supplied_rate_only"] for row in rows):
        raise ValueError("Only direct supplied-rate proposals are eligible")
    return [row["record"] for row in rows if row["record"] is not None]


def pooled_control(control, llm_root):
    old, new = read(control / "manifest.json"), read(llm_root / "manifest.json")
    keys = ("seed", "epochs", "batch_size", "rate_min", "rate_max", "quality_threshold", "evaluation_budget")
    if any(old["config"][k] != new["config"][k] for k in keys) or old["task_config"]["rpa"] != new["task_config"]["rpa"]:
        raise ValueError("Union task/seed/bounds/threshold/budget mismatch")
    audited = audit_self_feedback_bank(llm_root)
    bank, control_designs = direct_records(llm_root), designs(control)
    # Conservative charge includes repeated/cache-hit proposals: never understate work.
    charged = read(control / "completed.json")["counters"]["expensive_evaluations"] + len(bank)
    if charged > new["config"]["evaluation_budget"]:
        raise ValueError("Reused union exceeds the common cap; do not silently drop proposals")
    summary = topology_union(control_designs, bank, new["config"]["quality_threshold"])
    history = read(llm_root / "llm_history.json")
    return {"seed": new["config"]["seed"], **summary, "canonical_joint_charge_upper_bound": charged,
        "llm_direct_evaluation_upper_bound": len(bank), "audited_request_contexts": audited,
        "completed_rounds": sum(not x.get("error") for x in history),
        "failed_rounds": sum(bool(x.get("error")) for x in history),
        "control": str(control), "bank": str(llm_root),
        "bank_digest": hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest(),
        "type": "post-hoc-independent-union", "matched_completion_schedule": False}


def analyze(root):
    root = Path(root)
    plan = read(root / "plan.json")
    old_plan = read(CONTROL_ROOT / "active_plan.json")
    controls = {j["seed"]: Path(j.get("output", CONTROL_ROOT / "runs" / j["name"]))
                for j in old_plan["jobs"] if j["arm"] == "julia"}
    endpoints, unions, errors = [], [], []
    for job in plan["jobs"]:
        run = root / "runs" / job["name"]
        if not (run / "completed.json").exists():
            continue
        try:
            manifest = read(run / "manifest.json")
            if manifest["communication_mode"] != job["arm"] or manifest["config"]["seed"] != job["seed"]:
                raise ValueError("Endpoint identity mismatch")
            pooling = topology_union(designs(run), direct_records(run), manifest["config"]["quality_threshold"])
            endpoints.append({"seed": job["seed"], "arm": job["arm"], **pooling})
            if job["arm"] == "llm-to-rl":
                unions.append(pooled_control(controls[job["seed"]], run))
        except (KeyError, ValueError, OSError, sqlite3.Error) as error:
            errors.append({"seed": job["seed"], "arm": job["arm"], "error": str(error)})
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    result = {"endpoints": endpoints, "independent_unions": unions, "audit_errors": errors,
              "interpretation": "Descriptive reused baseline, not a matched asynchronous no-communication campaign. LLM inference cost remains attributable even though the bank is reused."}
    atomic_json(out / "summary.json", result)
    if unions:
        with (out / "independent_union.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(unions[0]))
            writer.writeheader()
            writer.writerows(unions)
    lines = ["# RPA Directional Campaign and Independent Union", "",
        f"Completed directional endpoints: {len(endpoints)}/40; audited reused unions: {len(unions)}/20.", "",
        "The reused baseline combines each historical AdaGrad/RL seed with that seed's direct supplied-rate LLM proposals from insertion-only. No fitted descendants or HOF-conditioned LLM proposals are used. Counts deduplicate complete structural hashes; lower loss replaces the same topology's previous rate setting.", "",
        "This is post-hoc terminal aggregation, not a newly executed no-communication arm. Model request counts, completion schedules, concurrency and numerical expenditures differ. Shared proposal banks make comparisons dependent; do not treat these as independent extra model runs or a fixed-proposal causal replay.", "",
        "| Endpoint | Complete | Mean qualifying topologies | Median best loss |",
        "|---|---:|---:|---:|"]
    for arm in ("llm-to-rl", "rl-to-llm", "independent-union"):
        rows = unions if arm == "independent-union" else [x for x in endpoints if x["arm"] == arm]
        if rows:
            lines.append(f"| {arm} | {len(rows)} | {statistics.mean(x['union_qualified'] for x in rows):.3f} | {statistics.median(x['union_best_loss'] for x in rows):.6g} |")
    lines += ["", "Incomplete results are descriptive only. A positive coupling difference is not attributable solely to policy imitation: mutation, archive selection and AdaGrad act on inserted candidates too.", "",
              f"Audit errors: {len(errors)}. Details and exact artifact paths are in summary.json."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
