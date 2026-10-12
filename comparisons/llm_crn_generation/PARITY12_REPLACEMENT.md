# Twelve-reaction parity replacement

User authorized stopping the eight-added-reaction cohort and replacing it on 2026-09-09.

Launch status: all four replacement arms started on 2026-09-09, with five active seeds per arm, 60 queued seeds, and Comet connected. Twenty new run databases and runtime configs were created; the new configs select twelve added reactions. The localhost viewer on port 8765 tracks this cohort using the same access token.

The original cohort is stopped, not invalidated: 20 completed runs (five matched seeds per arm), 20 interrupted runs, and 40 unstarted runs. All artifacts are preserved. Its status snapshot and stop reason are in `paper/iclr2027_genai_net_llm/generated/parity8_stopped_for_12_reactions_20260909.json`. No interrupted run is considered complete or a model failure. The completed eight-reaction cohort remains limited preliminary evidence, not a 20-seed comparison.

The connectivity check on its 20 loss-selected winners found a structural dependency path from every input species to OUT in every winner. Some auxiliary species were unused. Dependency edges link each mass-action reactant to species with nonzero net stoichiometric change, including consumption; this is potential influence, not proof of functional sensitivity. Increasing capacity is a hypothesis test, not a correction of an established disconnected-graph bug.

## Single scientific change

Exactly twelve added reactions, plus the four fixed input reactions (sixteen total), instead of eight added reactions. Species, indexed library, template mask, objective, CVODE settings, initial conditions, rate policy, model, prompts other than the count, and communication settings remain unchanged. Twelve reactions enlarge structural capacity but also increase search dimension; improvement is not guaranteed.

- Task variant: `parity-evidence-exploitation-12r-v1`.
- Separate config: `comparisons/rpa_search/configs/logic_parity_12reactions_300epoch.json`.
- Frozen review: `paper/iclr2027_genai_net_llm/generated/PARITY12_WORKSPACE_LOSS_PROMPTS.md`.
- Four arms, seeds 0-19, 300 epochs, batch 1023, candidate cap 307200.
- Five concurrent seeds per arm, four CVODE workers per seed, global Harness concurrency eight.
- DeepSeek V4 Flash; same Harness Decider/Writer and reasoning settings.
- Ten proposals every twenty epochs; initial HOF withheld; independent member validation.
- No exploratory evaluations, RAG, or optimization-exclusion mechanism.
- Comet required under `genai-net-llm-parity`.

The new method names, run suffixes, campaign roots, schedule, prompt approval and audit outputs all contain a separate twelve-reaction identifier. It starts fresh from seed zero without importing any eight-reaction HOF or model memory. The old default configuration and prompts remain unchanged. Never pool these two reaction budgets without explicitly distinguishing them.

## Launch

```bash
tmux new-session -d -s parity12-workspace-loss -c /local0/rossin/git/GenAI-Net 'env MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python comparisons/llm_crn_generation/run_parity12_communication_queue.py'
```

The queue refuses existing outputs; do not repeat the command after launch. It runs completion audits after all arms finish. Live status: `paper/iclr2027_genai_net_llm/generated/parity12_communication_300epoch_20260909_schedule.json`. Logs: the adjacent `parity12_20260909_logs/` directory. The localhost viewer tracks the replacement; the preserved eight-reaction records stay outside its active campaign plan. No new results are added to the manuscript before completion and analysis.
