# Four-input parity: versioned planned replacement

Update 2026-09-09: this eight-added-reaction campaign was stopped at the user's request, with five completed seeds per arm retained and active partial runs preserved. It was replaced by the isolated twelve-added-reaction campaign documented in `PARITY12_REPLACEMENT.md`. The launch details below describe the earlier campaign, not the currently running one.

Status: launched on 2026-09-08 in detached tmux session `parity-workspace-loss-v1`, following user authorization. All four arms started, with five active seeds per arm and Comet connected. Replaces the planned parity v1 cohort in the progress viewer. The old task prompt, shared system prompt, role templates, wrappers, and legacy queue commands remain available byte-for-byte; no historical run artifacts or manuscript prompt catalog were rewritten.

Live viewer: `http://127.0.0.1:8765/`. Its access token was rotated during the viewer-only restart; the private authenticated browser link is in `/local0/tmp/parity-viewer.html` (owner-readable, outside the repository). The launcher now resolves the repository import path itself, and its fallback database is the new full-duplex parity seed 0. The campaign page reloads the plan and live run status automatically; all 20 initial parity databases were verified selectable. The two preceding Flash Logic arms finished 20/20 seeds each with no failed runs before this launch. New parity progress is separate from those completed cohorts.

## Configuration

- Protocol: `workspace-loss-v1`; task prompt: `parity-evidence-exploitation-v1`.
- Four arms: full, LLM-to-RL, RL-to-LLM, none. Same prompt policy in every arm; communication still controls only the permitted information and insertion paths.
- DeepSeek V4 Flash via the existing Harness Decider/Writer processes, provider reasoning unchanged.
- 20 seeds per arm, 300 epochs, RL batch 1023, evaluation cap 307200, CVODE.
- Ten candidates per round, every 20 epochs, initial HOF withheld. Subsequent self-feedback remains available in all arms.
- No artificial workspace read/tool-call quota; one-hour stage timeout remains. Reading still consumes tokens and wall time. Only the current HOF overview is mandatory scientific context; task, role and validity constraints remain mandatory operational inputs.
- No exploratory-evaluation or literature service, skill, or endpoint is exposed. Normal post-submission canonical evaluation remains unchanged.
- Decider sees `DESIGN_CONSTRAINTS.md`, built from the actual evaluator contract, and the filtered library with allowed IDs/structures/parameter arities. No need to read the JSON schema.
- Six evidence-guided refinements plus four exploratory designs is a soft decision policy; absent usable parent evidence, exploratory proposals are permitted. No requirement to recover withheld context.
- Writer may resolve IDs and clamp finite rates to [0.001,100], but may not repair structures or invent rates. An unencodable member retains its original slot as `{"reaction_ids":[],"parameter_values":[]}`, with an indexed reason in `WRITER_ERRORS.json`. This is a deliberately invalid placeholder. Existing per-member validation rejects it and preserves valid siblings; an entirely invalid batch remains a failed outcome.
- Comet required for all runs. Existing release gates remain; an additional protocol-specific approval pins the new prompt-review hash.
- Parallelism unchanged: four arms, five concurrent seeds per arm, four CVODE workers per seed, global Harness concurrency eight. No new inference or simulation was used to configure the plan.

## Isolation and artifacts

New campaign prefix: `flash-parity4-v2-workspace-loss-`.
New run suffixes: `cvode_llm_flash_parity4_v2_workspace_loss_<arm>`.
Existing run directories are refused by the replacement queue; no automatic overwrite or resume.

Frozen prompts: `paper/iclr2027_genai_net_llm/generated/PARITY4_WORKSPACE_LOSS_V1_PROMPTS.md`.
Machine-readable prompt review and its conversation-authorized approval are adjacent JSON files.
Schedule: `paper/iclr2027_genai_net_llm/generated/parity4_communication_300epoch_workspace_loss_v1_schedule.json`.
The progress viewer's existing plan now points only its four planned parity rows at the new IDs. Current Flash Logic entries are unchanged.

Pre-launch validation command (the isolation guard now refuses it because campaign outputs exist):

```bash
MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python comparisons/llm_crn_generation/run_parity4_communication_300epoch_queue.py --protocol workspace-loss-v1 --plan-only
```

Launch command used from the repository root (do not repeat while running):

```bash
tmux new-session -d -s parity-workspace-loss-v1 -c /local0/rossin/git/GenAI-Net 'env MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python comparisons/llm_crn_generation/run_parity4_communication_300epoch_queue.py --protocol workspace-loss-v1'
```

The detached queue now supervises the four arms and performs the configured audits after completion. Inspect it with `tmux attach -t parity-workspace-loss-v1`; detach without stopping it using Ctrl-b, then d. Arm logs are under `paper/iclr2027_genai_net_llm/generated/parity4_workspace_loss_v1_logs/`. The previous parity configuration was planned but unlaunched; this substitutes that plan rather than creating an extra cohort.

## Scientific interpretation

This is a new protocol condition, not a retroactive change to older results. Within it, the four arms identify communication contrasts under the same evidence-use policy. Comparisons with older prompts also change reading policy, encoding guidance, and decision instructions; they cannot identify any one of those components in isolation. No new result has been added to the manuscript.
