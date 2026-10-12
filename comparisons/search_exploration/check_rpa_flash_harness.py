"""Isolated Harness readiness check, not an RL/AdaGrad campaign.

An actual paid Decider/Writer round requires --live. No credentials are copied
to artifacts; the existing DSH profile supplies authentication.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


PROMPT = Path(__file__).with_name("prompts") / "rpa-evidence-exploitation-v1.md"


def prepare(output, dsh_home):
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.llm.harness_client import HarnessLLMClient
    from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
    from RL4CRN.llm.workspace_loss_protocol import PROTOCOL

    config = json.loads(CONFIGS["rpa"].read_text())
    config["rpa"]["solver"] = "CVODE"
    config["search"]["rate_constant_range"] = [.001, 100.]
    evaluator, _, _ = build("rpa", config, .001, 100.)
    if evaluator.forbidden_reaction_ids != frozenset({0, 1, 28}):
        raise ValueError("Unexpected RPA null/template mask")
    if evaluator.max_added_reactions != 5:
        raise ValueError("Unexpected RPA reaction budget")
    client = HarnessLLMClient(
        workspace_root=output / "workspaces", dsh_home=dsh_home,
        provider="deepseek-official", model="deepseek-v4-flash",
        candidate_validation_policy="independent-members",
        harness_protocol=PROTOCOL,
    )
    graph = HarnessDeciderWriterCRNGraph(
        client=client, evaluator=evaluator, writer_retry_limit=0,
        max_workspace_evaluations=0, max_literature_searches=0,
    )
    return config, client, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsh-home", type=Path, default=Path.home() /
                        "ai-workspaces/deepseek-test/.dsh-home")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLCONFIGDIR", "/local0/tmp/mpl")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config, client, graph = prepare(output, args.dsh_home)
    task = PROMPT.read_text()
    manifest = {
        "kind": "readiness-only", "live": args.live,
        "provider": client.provider, "model": client.model,
        "protocol": client.harness_protocol, "command": list(client.command),
        "task_prompt": str(PROMPT.resolve()),
        "task_prompt_sha256": hashlib.sha256(task.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(client.system_prompt.encode()).hexdigest(),
        "task_config": config, "requested_candidates": 10,
        "withhold_initial_hof": True, "initial_llm_memory_empty": True,
        "exploratory_evaluations": 0, "rate_fitting": False,
        "training": False, "comet_upload": False,
        "note": "Later full-duplex callbacks in the async AdaGrad runner are not tested here.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.live:
        print(json.dumps({"status": "offline-preflight-only", **manifest}, indent=2))
        return
    try:
        result = graph.run_round(
            task_description=task, num_candidates=10, step=0,
            hall_of_fame_iter=None,
            forbidden_topologies_text="No topology exclusions supplied.",
            sil_feedback_text="Initial request: RL/HOF/SIL information intentionally withheld.",
            jsonl_path=output / "evaluations.jsonl",
        )
        workspace = client.last_workspace.path
        hof = json.loads((workspace / "CONTEXT/HALL_OF_FAME.json").read_text())
        if hof["entries"] or (workspace / "CONTEXT/hall-of-fame").exists():
            raise RuntimeError("Initial HOF exposure detected")
        if client.last_workspace.call_count != 2:
            raise RuntimeError("Expected exactly one Decider and one Writer invocation")
        summary = {
            "status": "completed", "workspace": str(workspace),
            "elapsed_seconds": time.monotonic() - started,
            "model": client.model, "role_invocations": client.last_workspace.call_count,
            "accepted_candidates": len(result.evaluations),
            "valid_evaluations": sum(bool(e.valid) for e in result.evaluations),
            "losses": [e.loss for e in result.evaluations],
            "response_validation": result.response_validation,
            "initial_hof_entries": len(hof["entries"]),
            "exploratory_evaluations": len(result.tool_evaluations),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        failure = {"status": "failed", "type": type(exc).__name__,
                   "elapsed_seconds": time.monotonic() - started,
                   "workspace": str(client.last_workspace.path) if client.last_workspace else None}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
