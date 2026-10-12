# Paired parity-12 evolutionary replay

Campaign: `flash-parity12-insertion-fixed-300epoch-20260909`.

This follow-up waits for **all 40 current Logic GA runs** and **all 20 Flash
parity-12 insertion-only source runs** to finish. Source audit and code fingerprints
must pass before any search starts. The active campaigns and their code are not
changed. A failed dependency prevents launch; a failed GA seed stops further
dispatch, preserves artifacts, and requires inspection rather than automatic retry.

## Paired design

- Four-input parity, **12 added reactions plus four fixed input reactions**.
- Seeds 0--19, 300 batches of 1,023: **306,900 search evaluations per seed**.
- Replay each source run's actual evaluated Flash proposals, unchanged, at its
  recorded completion-epoch boundary. No new LLM calls or substitute proposals.
- The total per-seed budget is 306,900 plus the source's actual evaluated LLM
  proposals (at most 150); the configured 307,200 cap is not extra search work.
- Late terminal replies enter only terminal aggregation and cannot generate offspring.
- Saved HOF contexts must be empty for every insertion-only request. The source
  LLM had self-feedback but no RL/HOF conditioning, including request zero.
- Same canonical CVODE evaluator, all 16 truth-table rows, loss weights, initial
  conditions, reaction library/mask, and top-30 topology archive as the source.
- Same mutation-only evolutionary search as the Logic GA: 45% single-rate
  mutation, 45% single-reaction replacement, 10% random proposals; no crossover.
- Four concurrent runs, four CPU workers each, normal priority, no GPU use.
- Comet project: `genai-net-ga-fixed-proposals`. Credentials are read from the
  existing configuration, never stored in these files.

Pairing controls proposals, availability in search-evaluation units, and evaluator
counts, **not elapsed time, random populations, or total compute cost**. Mutation
rates are positive and unbounded as in the existing evolutionary baseline; source
LLM rates are replayed exactly. Initial/replacement rates use the existing [0.1, 50]
prior. These are explicit optimizer differences, not an identical RL search policy.

## Outcomes and audit

Primary comparison: paired canonical endpoint loss. A separate functional
diagnostic counts distinct topologies in each loss-ranked top-30 terminal archive
that classify all 16 supplied rows correctly (`OUT(t=100) >= 0.5`). Incomplete,
nonfinite, or canonical failure-fill trajectories are ineligible for correctness;
their canonical failure losses remain in search accounting. No Logic loss cutoff
is imported. This is training-table correctness, not robustness or generalization.

Every injected proposal's loss and topology hash must reproduce its saved values.
After search, the 30 source terminal designs are reevaluated solely to verify their
losses and obtain comparable correctness diagnostics. These 30 verification calls
are recorded separately and never inform the GA or consume its search budget.

Per seed: `candidates.csv`, `progress.csv`, `terminal_archive.json`,
`source_archive_validation.json`, `completed.json` (or `failed.json`), frozen
`replay.json`, `config.json`, and `comet.json`. Candidate records retain parent
hashes and inherited proposal IDs for subsequent ancestry analysis.

## Queue

From the repository root:

```bash
# Prepare once; refuses to overwrite an existing campaign.
.venv/bin/python -m comparisons.llm_seeded_evolution.parity12.queue

# Read dependency counts, without launching anything.
.venv/bin/python -m comparisons.llm_seeded_evolution.parity12.queue --check-once

# The detached ga-parity12-followup tmux session runs this watcher.
env -u COMET_API_KEY CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/local0/tmp/mpl \
  .venv/bin/python -m comparisons.llm_seeded_evolution.parity12.queue --watch
```

Queue artifacts live in `campaigns/flash-parity12-insertion-fixed-300epoch-20260909/`
under this folder. Inspect `status.json` and `watcher.log`. Do not start a second
watcher or change frozen implementation files while it is scheduled.
