"""Run small live Harness proposal rounds against the MMC2 benchmark evaluators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper
from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
from RL4CRN.llm import (
    HarnessCRNGenerator,
    HarnessLLMClient,
    LLMCandidateEvaluator,
    get_mmc2_task_prompt,
)
from comparisons.rpa_search.src.common.logic_task import build_logic_components
from comparisons.rpa_search.src.common.rpa_task import build_rpa_components
from comparisons.llm_crn_generation.paper_breadth_tasks import (
    BREADTH_BUILDERS,
    BREADTH_TASKS,
)


CONFIGS = {
    "logic": REPO_ROOT / "comparisons/rpa_search/configs/logic_100k.json",
    "rpa": REPO_ROOT / "comparisons/rpa_search/configs/rpa_100k.json",
}
CONFIGS.update(
    {
        task: REPO_ROOT
        / "comparisons/llm_crn_generation/configs/paper_breadth_100epoch.json"
        for task in BREADTH_TASKS
    }
)
BUILDERS = {
    "logic": build_logic_components,
    "rpa": build_rpa_components,
}
BUILDERS.update(BREADTH_BUILDERS)


def build_evaluator(task_name: str, *, solver: str | None = None) -> LLMCandidateEvaluator:
    config = json.loads(CONFIGS[task_name].read_text(encoding="utf-8"))
    if solver is not None:
        config[task_name]["solver"] = solver.strip().upper()
    crn, library_components, task, _ = BUILDERS[task_name](config)
    library = library_components[0]
    search = config["search"]
    rate_min, rate_max = search["rate_constant_range"]
    null_reaction_id = library.find_zero_reaction()
    return LLMCandidateEvaluator(
        crn_template=crn,
        max_added_reactions=int(search["max_added_reactions"]),
        library=library,
        stepper=IOCRNStepper(),
        actuator=LibraryActuator(library),
        compute_reward_func=task.compute_reward,
        min_parameter_value=float(rate_min),
        max_parameter_value=float(rate_max),
        enforce_parameter_bounds=True,
        forbidden_reaction_ids=[null_reaction_id],
    )


def run_task(
    task_name: str,
    *,
    num_candidates: int,
    workspace_root: Path,
    dsh_home: Path,
    provider: str,
    model: str,
    llm_base_url: str | None,
    timeout_seconds: float,
) -> dict:
    client = HarnessLLMClient(
        workspace_root=workspace_root,
        dsh_home=dsh_home,
        provider=provider,
        model=model,
        openai_compatible_base_url=llm_base_url,
        timeout_seconds=timeout_seconds,
    )
    generator = HarnessCRNGenerator(
        client=client,
        evaluator=build_evaluator(task_name),
    )
    result = generator.run_round(
        task_description=get_mmc2_task_prompt(task_name),
        num_candidates=num_candidates,
    )
    return {
        "task": task_name,
        "model": client.model,
        "workspace": str(client.last_workspace.path),
        "candidate_count": len(result.evaluations),
        "valid_count": sum(evaluation.valid for evaluation in result.evaluations),
        "losses": [evaluation.loss for evaluation in result.evaluations],
        "messages": [evaluation.message for evaluation in result.evaluations],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--llm-provider", default="deepseek-official")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/crn-runs",
    )
    parser.add_argument(
        "--dsh-home",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/.dsh-home",
    )
    args = parser.parse_args()
    task_names = list(CONFIGS) if args.task == "all" else [args.task]
    summaries = [
        run_task(
            task_name,
            num_candidates=args.num_candidates,
            workspace_root=args.workspace_root,
            dsh_home=args.dsh_home,
            provider=args.llm_provider,
            model=args.model,
            llm_base_url=args.llm_base_url,
            timeout_seconds=args.timeout,
        )
        for task_name in task_names
    ]
    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
