# DeepSeek Harness CRN backend

`HarnessLLMClient` runs the DeepSeek bot as a separate headless process only
when a generation round is requested. The bot's current directory is a new,
Git-initialized run directory and not the repository.

## Notebook setup

```python
from pathlib import Path

from RL4CRN.llm import HarnessCRNGenerator, HarnessLLMClient

client = HarnessLLMClient(
    workspace_root=Path.home() / "ai-workspaces/deepseek-test/crn-runs",
    dsh_home=Path.home() / "ai-workspaces/deepseek-test/.dsh-home",
)

proposal_graph = HarnessCRNGenerator(
    client=client,
    evaluator=evaluator,
    spec=graph_spec,
)
```

The API key is read by Harness from its own configuration. Do not pass it to
Python or place it in the repository.

The client writes a per-run Harness patch that sets the actual Harness system
persona and pins `deepseek-official/deepseek-v4-flash`. The shared persona and
the canonical logic/RPA task prompts live in
`RL4CRN/llm/benchmark_prompts.py`.

Run one candidate for each MMC2 benchmark with:

```bash
MPLCONFIGDIR=/tmp/rl4crn-mpl .venv/bin/python \
  comparisons/llm_crn_generation/run_mmc2_harness_smoke.py \
  --task all --num-candidates 1
```

Run the asynchronous RL4CRN + Harness hybrid benchmark with 64 rollout workers
and ten proposals per completed Harness round:

```bash
MPLCONFIGDIR=/tmp/rl4crn-mpl .venv/bin/python \
  comparisons/llm_crn_generation/run_mmc2_harness_hybrid.py \
  --task all --runs 3 --epochs 100 --n-cpus 64 \
  --llm-candidates 10 --llm-every 1 --max-agent-evaluations 10
```

Each completed proposal round uses one Harness/model request that returns all ten
candidates in one JSON payload. The reaction library and live search state are
read from workspace files instead of being duplicated in the request text.

Harness generation runs in a background thread and a separate headless process.
RL epochs continue while the model is pending. At an epoch boundary, completed
candidates are merged on the RL thread into the shared Hall of Fame before SIL.

The pinned Harness package is installed once under
`~/ai-workspaces/deepseek-test/dsh-runtime/` and invoked directly. Requests do
not run `npx` or `pnpm dlx`, avoiding package-store contention under parallel
campaigns. A filesystem-backed gate under `DSH_HOME/request-slots/` bounds
inference globally across independent seed processes. Hosted and local providers
use separate pools because their backends do not contend. Each slot owns one
reusable provider-scoped `headless-worker-<provider>-NNN` profile clone; request
patches and workspaces remain isolated. Waiting for a clone occurs only in the
background LLM thread and never blocks RL/CVODE progress. Hosted campaigns use
eight clones, while a single local-model clone matches the one-GPU server.

## Per-run files

Each `run_round(...)` call creates one directory under the dedicated
`~/ai-workspaces/deepseek-test/crn-runs/` tree with:

- `TASK.md`: the scientific task given to the bot.
- `SYSTEM_PROMPT.md`: the shared operational behavior for every CRN task.
- `harness.patch.yml`: the pinned provider/model and actual Harness system persona.
- `OUTPUT_CONTRACT.json`: exact output schema, reaction IDs, and parameter arities.
- `CONTEXT/`: per-round Hall of Fame, SIL status, excluded topologies, search state,
  and cached trajectory diagnostics/plots extracted from the live run without new simulations.
- `.dsh/skills/crn-simulation/SKILL.md`: instructions for inspecting cached evidence first and
  optionally invoking the capped CVODE evaluator.
- `REASONING_NOTES.md`: concise scientific rationale, evidence used, decisions, and measured outcomes.
- `tool-evaluations.jsonl`: separately counted exploratory evaluator calls made by the agent.
- `calls/`: prompts plus stdout, stderr, model, duration, and exit status for each bot process.
- `candidates.json`: the parsed proposals.
- `evaluations.jsonl`: format checks, simulator losses, and CRN strings.
- `evaluation_summary.json`: valid/invalid counts and the best loss.
- `run_status.json`: completed or failed status.

`run_manifest.json` records the pinned provider/model and SHA-256 hashes of the
system prompt, task prompt, and output contract for paper reproducibility.

The task, contract, and manifest are committed as the run repository's baseline
before Harness starts. Model-created files and evaluation artifacts remain as
reviewable Git changes.

The Python contract validator rejects wrong candidate counts, extra fields,
unknown or duplicate reaction IDs, wrong parameter-vector lengths, and
non-positive or non-finite parameter values before simulation. Valid candidates
are then evaluated through `LLMCandidateEvaluator`, using the same RL4CRN
actuator, stepper, task, and Hall of Fame as the existing backends.

Set `RL4CRN_DSH_COMMAND` only if automatic Harness discovery does not find the
pinned workspace installation. Falling back to a package-manager launcher is
supported for setup and smoke testing, but should not be used for parallel
campaigns.
