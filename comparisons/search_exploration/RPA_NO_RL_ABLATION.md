# RPA Learned-Emitter Ablation

## Intervention

Reference: `/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/run`.
Control: `/local1/rossin/search-exploration/rpa-streams-no-rl-100epoch-20260915/run`.

The reference uses 50% learned structural proposals, 25% single-reaction archive
mutation and 25% random proposals. The control replaces the learned share with
random proposals: 0% RL, 25% archive mutation, 75% random. Empty-archive mutation
falls back to random in both arms. Policy-gradient and elite-imitation updates
are skipped entirely in the control. An unused initialized policy remains in the
session for checkpoint/interface compatibility; it generates no candidates.

Seed 0, 100 epochs, batch 1023, 102300 proposal attempts, evaluation cap 1636800,
canonical RPA/CVODE evaluator, 16-call evolutionary rate fitting, bounds
[0.001, 100], archive 2000 and corrected v2 retention are unchanged. The control
uses the reference's exact frozen task/configuration, except `rl_fraction`.
Worker split remains 32 search / 16 optimization, queue 512, up to 256 selected
fits per epoch. There are no LLM calls. Comet records metrics and existing plots.

This isolates removal/replacement of the learned structural emitter, not a pure
policy-gradient-loss ablation or a tuned standalone-GA comparison. Mutation share
is not increased to compensate for RL removal. The asynchronous queue can admit
different numbers of fits when policy compute is absent. Report actual evaluations
and elapsed time as well as epoch count; consumed evaluations and completion
schedules are not exactly matched. Random sequences diverge after the intervention.

## Execution

Launched in tmux `rpa-no-rl-control-20260915`:

```bash
env -u COMET_API_KEY CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/local0/tmp/mpl \
  .venv/bin/python -m comparisons.search_exploration.run_no_rl_control \
  --reference /local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/run \
  --output /local1/rossin/search-exploration/rpa-streams-no-rl-100epoch-20260915
```

Existing output directories are refused, not overwritten. No retries or automatic
replacement of failures. The controller writes progress and, after all epochs and
optimizer draining, generates comparison plots/report plus source attribution.
`run/completed.json` confirms the experiment; root `completed.json` also confirms
postprocessing. Absence of these files means completion is not established.

Comet: https://www.comet.com/redsnic/genai-net-exploration/92089b8f53cc453284c62a284ec31a05

## Existing Interaction Evidence

From the completed corrected-v2 reference's chronological event ledger:

| First qualification pathway | Distinct topologies |
| --- | ---: |
| Direct RL proposals, before their own rate fit | 139 |
| RL proposals after rate optimization | 30 |
| Archive mutation at inherited initial rates | 7 |
| Random proposals, before or after fitting | 0 |

The counts sum to 176 and assign first qualification in recorded merge order.
They are not source-normalized success rates. Later cross-source rediscoveries
are not credited as additional first discoveries.

Of 25328 direct archive-mutation events, 18749 have RL origin and 6579 random
origin. All 8 qualifying mutation events have RL origin (8 topologies seen;
7 were first qualifications, with overlap against the RL stream). All mutation
events change the structural topology hash. The provenance tracks explicit
parentage, not influence through learned policy weights. It demonstrates actual
reuse of RL designs by mutation, but not causal synergy or an identified
GA-to-RL learning benefit. The shared imitation archive provides the reverse
pathway; the current logs do not identify which demonstrations caused later draws.

Reproducible analysis and caveats:
`reports/rpa-streams-v2-source-interaction-20260915/SOURCE_INTERACTION.md`.

For a stronger interaction claim, replicated seeds and reciprocal removal of
archive mutation would be needed; neither is claimed to have been run here.
