"""Shared agent instructions and paper-benchmark CRN task prompts."""

from __future__ import annotations


CRN_AGENT_SYSTEM_PROMPT = """You are an expert operator in a two-stage constrained chemical reaction network design workflow.

Operational rules:
1. Work only inside the current run workspace. Never inspect or modify parent directories, unrelated files, Harness configuration, credentials, or source repositories.
2. Do not invoke Bash, install dependencies, change machine configuration, use the network, or run destructive commands. Use the Harness native read, write, edit, glob, and grep tools for workspace access. The external RL4CRN evaluator is the sole authority for validation and loss. For optional exploratory simulations, use only the capped workspace queue described by the crn-simulation skill. When and only when a literature-search skill is present, you may also use its capped read-only local corpus queue.
3. Read TASK.md, all current files under CONTEXT/, and the current call request before answering. The Writer must also read OUTPUT_GUIDE.json and treat it as a hard encoding requirement; the Decider must not read the JSON guide. OUTPUT_CONTRACT.json remains the host validator's authoritative source and neither role may read it directly. Do not page through the reaction library or repeatedly announce searches: use targeted grep queries on REACTION_LIBRARY.tsv for concrete reactant/product patterns. Dynamic Hall-of-Fame, SIL, and excluded-topology state belongs in CONTEXT/ and must not be copied into a growing prompt.
4. Select reactions only from the supplied ID-indexed library. Never invent reaction IDs, species, reactions, parameter slots, evaluation results, or scientific claims.
5. Obey the role named by the current call. The Decider owns the scientific choices and must specify concrete CRN structures and intended rates; it may use equations, names, IDs, tables, or another concise notation and is not required to emit JSON. The Writer must preserve those choices, implement the constraints, resolve them to supplied library IDs, and encode the exact JSON object. The Writer must not become a second scientific proposer.
6. The direct-LLM rate interval is [0.001, 100]. A finite Decider rate below 0.001 is truncated to 0.001 and one above 100 is truncated to 100 by the Writer and host. Non-numeric and non-finite values are invalid. The Decider may state an out-of-range intended rate when scientifically useful, but must account for the deterministic truncation.
7. When more than one candidate is requested, balance exploration and refinement: propose new reaction sets and retune promising admissible Hall-of-Fame sets. Never repeat an identical topology-and-parameter candidate. A Hall-of-Fame topology may be refined unless it is separately listed in the forbidden-topology archive.
8. Do not claim that a candidate works before evaluation. Aim to provide the requested number, but each Writer member is validated independently: an invalid member is rejected without removing unrelated valid members and is not silently replaced. Before encoding, verify each member's reaction budget, unique allowed IDs, parameter-vector count and arity, finite rates, and final clamped bounds.
9. Return only the response format requested by the current call. For a JSON call, use no markdown fences or surrounding commentary, write the exact object to FINAL_RESPONSE.json as the final workspace action, and return the identical object. For a Decider call, return a concise readable design record rather than private chain-of-thought.
10. Work iteratively across scheduled rounds. Prefer proposing the requested diverse batch promptly and letting the external evaluator score it. Exploratory simulation is exceptional: use it only when one concrete uncertainty could materially change the current batch, never to pre-screen every proposal. Finish each stage after at most eight workspace-tool calls; if evidence is sparse, make concrete diverse choices under uncertainty and wait for external results.
"""


MMC2_LOGIC_TASK_PROMPT_REPORTED_2026 = """Design candidate mass-action CRNs for the paper's MMC2 logic-circuit benchmark.

Fixed benchmark definition:
- Inputs: four binary signals u_1, u_2, u_3, u_4.
- Template species: X_1, X_2, X_3, X_4 and output species OUT.
- Fixed template reactions: each u_i drives production of X_i. The benchmark has no support species and no fixed dilution reactions.
- Desired Boolean function: (u_1 AND u_2) OR (u_2 AND u_3) OR (u_3 AND u_4).
- Evaluation scenarios: all 16 vectors in {0, 1}^4.
- Initial concentration: 0.01 for every species.
- Simulation: LSODA over t = 0 to 100 with 1000 time points, rtol = atol = 1e-8.
- Loss: mean binary cross-entropy between the desired Boolean value and the final OUT concentration over all scenarios. Lower is better.
- Search space: choose exactly five distinct reactions from the supplied order-2 mass-action library.
- The null emptyset-to-emptyset reaction is inadmissible.
- Each selected reaction has one scalar rate constant in the inclusive interval [0.1, 50.0].

Prefer candidates that keep OUT close to zero for every false row and close to one for every true row, remain finite over the full simulation, and use complementary mechanisms rather than trivial parameter variants of one topology. The supplied reaction IDs and output contract are authoritative.
"""


# The reported 2026 campaigns used the prompt above, whose BCE description did
# not match LogicTaskKind's canonical trajectory objective. Keep it available
# for artifact reproduction while using the corrected contract in future runs.
MMC2_LOGIC_TASK_PROMPT = MMC2_LOGIC_TASK_PROMPT_REPORTED_2026.replace(
    "Loss: mean binary cross-entropy between the desired Boolean value and the final OUT concentration over all scenarios. Lower is better.",
    "Loss: transient-weighted mean absolute error between the desired Boolean value and the full OUT trajectory over all scenarios. The first 20% of each trajectory has weight 0.25, the middle 60% weight 1, and the final 20% weight 2. Lower is better.",
)


MMC2_LOGIC_TRAJECTORY_TASK_PROMPT = """Design candidate mass-action CRNs for the paper's MMC2 logic-circuit benchmark.

Fixed benchmark definition:
- Inputs: four binary signals u_1, u_2, u_3, u_4.
- Template species: X_1, X_2, X_3, X_4 and output species OUT.
- Fixed template reactions: each u_i drives production of X_i. The benchmark has no support species and no fixed dilution reactions.
- Desired Boolean function: (u_1 AND u_2) OR (u_2 AND u_3) OR (u_3 AND u_4).
- Evaluation scenarios: all 16 vectors in {0, 1}^4.
- Initial concentration: 0.01 for every species.
- Simulation: LSODA over t = 0 to 100 with 1000 time points, rtol = atol = 1e-8.
- Loss: output-trajectory L1 error over all 16 rows. Time weights are 0.25 for the first 20% of each trajectory, 1.0 for the middle 60%, and 2.0 for the final 20%. Lower is better.
- Search space: choose exactly five distinct reactions from the supplied order-2 mass-action library.
- The null emptyset-to-emptyset reaction is inadmissible.
- Each selected reaction has one scalar rate constant in the inclusive interval [0.1, 50.0].

Keep OUT close to zero on every false row and close to one on every true row. Transient deviations are also charged by the executable loss, with later time points weighted most strongly. Prefer finite trajectories and complementary mechanisms rather than trivial parameter variants of one topology. The supplied reaction IDs, output contract, and external evaluator are authoritative.
"""


MMC2_RPA_TASK_PROMPT = """Design candidate mass-action CRNs for the paper's MMC2 robust-perfect-adaptation tracking benchmark.

Fixed benchmark definition:
- Species: X_1, X_2, X_3.
- Fixed template input reaction: u_1 drives production of X_1.
- Fixed template disturbance reaction: u_2 drives degradation of X_3.
- Measured output: X_3.
- Tracking target: X_3 should track u_1 while being insensitive to the disturbance u_2.
- Evaluation scenarios: the Cartesian product u_1, u_2 in {0.5, 1.0, 1.5}, for nine scenarios.
- Initial concentration: 0.01 for every species.
- Simulation: LSODA over t = 0 to 100 with 1000 time points, rtol = atol = 1e-8.
- Loss: the benchmark transient tracking loss against target r = u_1 over all scenarios. Lower is better.
- Search space: choose exactly five distinct reactions from the supplied order-2 mass-action library.
- The null emptyset-to-emptyset reaction is inadmissible.
- Each selected reaction has one scalar rate constant in the inclusive interval [0.1, 50.0].

Prefer mechanistically plausible feedback, integral-like, or buffering motifs that can reject changes in u_2 without preventing X_3 from tracking changes in u_1. Avoid disconnected reactions and candidates that merely tune one disturbance level. The supplied reaction IDs and output contract are authoritative.
"""


MMC2_TASK_PROMPTS = {
    "logic": MMC2_LOGIC_TASK_PROMPT,
    "rpa": MMC2_RPA_TASK_PROMPT,
}

_BREADTH_COMMON = """

Shared extension protocol:
- Select exactly six distinct reactions from the supplied order-2 mass-action library.
- The null reaction is inadmissible and every rate constant must be in [0.1, 50.0].
- Treat the supplied output contract and external evaluator as authoritative; lower loss is better.
"""

MMC2_TASK_PROMPTS.update(
    {
        "dose_hill": """Design a three-species CRN whose X_3 output follows the frozen Hill target 2*u_1^2/(0.25^2+u_1^2) over ten input levels from 0 to 1. The fixed input reaction produces X_1. Evaluation uses weighted transient tracking over t=0..100 with CVODE.""" + _BREADTH_COMMON,
        "dose_ultrasensitive": """Design a three-species CRN whose X_3 output follows the frozen ultrasensitive target 2*u_1^8/(0.5^8+u_1^8) over ten input levels from 0 to 1. The fixed input reaction produces X_1. Evaluation uses weighted transient tracking over t=0..100 with CVODE.""" + _BREADTH_COMMON,
        "dose_biphasic": """Design a three-species CRN whose X_3 output follows the frozen non-monotonic target 8*u_1/(1+u_1)/(1+(u_1/0.55)^4) over ten input levels from 0 to 1. The fixed input reaction produces X_1. Evaluation uses weighted transient tracking over t=0..100 with CVODE.""" + _BREADTH_COMMON,
        "classifier": """Design a two-species autonomous fate-decision CRN. Across 32 labeled initial conditions, trajectories must converge to [1,0] or [0,1] according to the supplied diagonal and clustered classes. Evaluation uses time-integrated L1 tracking over t=0..100 with CVODE.""" + _BREADTH_COMMON,
        "oscillator_mean": """Design an autonomous three-species CRN with sustained oscillations in output X_3 around fixed temporal mean 1. Evaluation combines periodicity, damping, and mean error over t=0..100 with CVODE.""" + _BREADTH_COMMON,
        "oscillator_frequency": """Design a three-species CRN whose X_3 output sustains oscillations with frequency controlled by u_1. Tested target frequencies are 0.1, 1/15, and 0.05. Evaluation combines periodicity, frequency error, and damping over t=0..100 with CVODE.""" + _BREADTH_COMMON,
        "stochastic_rpa": """Design a four-species stochastic RPA CRN with controller species Z_1,Z_2,Z_3 and output X_1. Input u_1 drives Z_1 production and disturbance u_2 drives X_1 degradation. Across u_1,u_2 in {0.5,1,1.5}, track mean target 3*u_1 while reducing coefficient of variation. Evaluation uses 1000 SSA trajectories per condition; CVODE is retained only for deterministic post-analysis.""" + _BREADTH_COMMON,
    }
)

# Exact task prompts used by the frozen paper campaigns. This mapping prevents
# regenerating the reproducibility catalog with a post-audit correction.
MMC2_REPORTED_TASK_PROMPTS_2026 = dict(MMC2_TASK_PROMPTS)
MMC2_REPORTED_TASK_PROMPTS_2026["logic"] = MMC2_LOGIC_TASK_PROMPT_REPORTED_2026


# Contract-v2 widens direct-LLM rate support. The historical mapping above is a
# byte-preserving snapshot and must remain unchanged for forensic reproduction.
_OLD_RATE_RULE = "[0.1, 50.0]"
_CONTRACT_V2_RATE_RULE = "[0.001, 100]"


def _with_contract_v2_rate_bounds(prompt: str) -> str:
    return prompt.replace(_OLD_RATE_RULE, _CONTRACT_V2_RATE_RULE)


MMC2_TASK_PROMPTS = {
    task: _with_contract_v2_rate_bounds(prompt)
    for task, prompt in MMC2_TASK_PROMPTS.items()
}
MMC2_LOGIC_TASK_PROMPT = MMC2_TASK_PROMPTS["logic"]
MMC2_RPA_TASK_PROMPT = MMC2_TASK_PROMPTS["rpa"]
MMC2_LOGIC_TRAJECTORY_TASK_PROMPT = _with_contract_v2_rate_bounds(
    MMC2_LOGIC_TRAJECTORY_TASK_PROMPT
)


def _get_task_prompt(
    prompts: dict[str, str], task: str, *, solver: str | None = None
) -> str:

    try:
        prompt = prompts[task.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(prompts))
        raise ValueError(f"Unknown MMC2 task {task!r}; choose one of: {choices}.") from exc
    if solver is None:
        return prompt
    normalized_solver = solver.strip().upper()
    if normalized_solver not in {"CVODE", "LSODA"}:
        raise ValueError("solver must be either 'CVODE' or 'LSODA'.")
    return prompt.replace("Simulation: LSODA", f"Simulation: {normalized_solver}")


def get_mmc2_task_prompt(task: str, *, solver: str | None = None) -> str:
    """Return the corrected task prompt for new runs."""

    return _get_task_prompt(MMC2_TASK_PROMPTS, task, solver=solver)


def get_reported_mmc2_task_prompt_2026(
    task: str, *, solver: str | None = None
) -> str:
    """Return the exact task prompt used by the frozen 2026 campaigns."""

    return _get_task_prompt(MMC2_REPORTED_TASK_PROMPTS_2026, task, solver=solver)


def get_mmc2_task_prompt_variant(
    task: str,
    *,
    variant: str = "standard",
    solver: str | None = None,
) -> str:
    """Return an explicitly named campaign prompt without mutating frozen prompts."""

    normalized = variant.strip().lower()
    if normalized == "standard":
        return get_mmc2_task_prompt(task, solver=solver)
    if normalized == "reported-2026":
        return get_reported_mmc2_task_prompt_2026(task, solver=solver)
    if normalized == "logic-trajectory":
        if task.strip().lower() != "logic":
            raise ValueError("The logic-trajectory prompt variant is valid only for task='logic'.")
        return _get_task_prompt(
            {"logic": MMC2_LOGIC_TRAJECTORY_TASK_PROMPT},
            "logic",
            solver=solver,
        )
    raise ValueError(
        "Unknown task prompt variant; choose one of: logic-trajectory, "
        "reported-2026, standard."
    )
