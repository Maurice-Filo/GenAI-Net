# Opt-in topology exploration

This is a new implementation, not a modification of the legacy experiments.
Its settings are engineering pilot defaults, not an approved scientific protocol.
No existing campaign, prompt, result, or manuscript is rewritten.

## Matched RPA workflow campaign

The separate [frozen statistical protocol](RPA_STATISTICAL_PLAN.md) compares
Julia-fit hybrid, evolution-fit hybrid and bounded standalone GA on 20 matched
RPA seeds. It specifies useful-topology discovery yield, paired endpoint loss,
four-test multiplicity correction, failure handling and unequal-compute caveats.
The seed-0 engineering trials are not part of this new inference cohort.
`rpa_statistical_campaign.py` schedules at most 64 evaluator/fitter workers;
`analyze_rpa_statistical.py` produces local reports and PNG/PDF endpoint figures.
No LLM calls, pruning or external uploads are enabled in this campaign.

The 2026-09-16 user-authorized amendment disables scientific evaluation/fitting
wall-clock limits for remaining jobs. See the campaign's
`amendments/no-timeouts/AMENDMENT.md` and `active_plan.json`; the original plan
and failed attempt are retained. Four completed, non-time-limited results carry
forward. The failed GA seed restarts separately, with aborted-attempt compute
reported as overhead. Startup/shutdown health checks and numerical validity
checks remain. The common attempt ceiling does not imply identical expenditure.

The subsequent concurrency amendment permits two hybrids and two GAs at once,
up to 128 worker slots on 128 hardware threads. Per-run settings are unchanged;
load can affect runtime and asynchronous fitting admission. The live state is
selected by `active_plan.json`: `progress-concurrent.json`, with the current
report at `analysis-concurrent/REPORT.md`. The original controller only drains
its active pair and reports their real exit statuses for adoption. Use
`HOLD_CONCURRENT` to stop new dispatch in the current controller.

After the disk-full interruption, the campaign's physical storage is
`/local1/rossin/search-exploration/rpa-statistical-20seed-20260915`; its original
path under `reports/` is a compatibility symlink. Twenty-six validated completed
runs were retained and four interrupted arm-seed pairs restarted separately.
See [RPA_DISK_RECOVERY.md](RPA_DISK_RECOVERY.md) and the campaign's
`amendments/disk-recovery/` for the audit, preserved attempts and launch record.
The active tmux session is `rpa-statistical-local1-20260916`.

The completed cohort's post-hoc quality/diversity/mechanism atlas is in
`reports/rpa-statistical-20seed-20260915/analysis-mechanisms-20260917/REPORT.md`,
with `RPA_PORTFOLIO_ATLAS.pdf` beside it. Reproduce with
`python -m comparisons.search_exploration.analyze_rpa_mechanisms ROOT --output OUTPUT`.
This reads saved results and cached trajectories only; it does not launch
simulations or change the manuscript. Structural families are exploratory,
not validated biochemical mechanisms; equal-size comparisons explicitly restrict
to eligible matched seeds.

The separate completed rate-perturbation study is at
`/local1/rossin/search-exploration/rpa-rate-perturbations-20260917/analysis/REPORT.md`,
with `RPA_PERTURBATIONS.pdf`, five vector/PNG figures and CSV/JSON backing data.
It evaluates the 60 current saved winners and 19 historical RL-only winners;
historical seed 10 is explicitly excluded for a null reaction, not replaced.
There are 18,309 canonical evaluations plus 651 failure-diagnostic replays.
No rates are refitted or clipped, and no original campaign is modified.

```bash
.venv/bin/python -m comparisons.search_exploration.run_rpa_perturbations --prepare --campaign CAMPAIGN --output NEW_OUTPUT --workers 64
.venv/bin/python -m comparisons.search_exploration.run_rpa_perturbations --execute --output NEW_OUTPUT
.venv/bin/python -m comparisons.search_exploration.audit_rpa_perturbation_failures NEW_OUTPUT
.venv/bin/python -m comparisons.search_exploration.analyze_rpa_perturbations NEW_OUTPUT
```

Preparation requires a new directory and freezes input/source hashes. Execution
resumes only missing jobs. The failure audit never replaces original outcomes.
The optional balance-preserving control holds sensing/calibration terms fixed;
it varies fewer parameters and is not an equal-dimensional robustness comparison.
Historical search settings differ, so method contrasts remain descriptive.

## Workflow

1. Generate a fixed-policy cohort from RL, archive mutation, and random sources.
2. Fit rates using the same canonical CVODE evaluator, exact evaluation cache,
   declared rate bounds, initializations, evaluation cap, and deadline.
3. Commit outcomes in proposal order. Calculate archive gain and decreasing
   region-visit novelty separately from physical loss.
4. Update the structural encoder/head with on-policy terminal rewards, structural
   entropy and top-elite imitation. Rate heads receive no gradient.
5. Checkpoint, persist conventional HOF snapshots, and log metrics/plots.

Optional Harness calls retain the existing Decider/Writer implementation,
independent-member validation, separate workspace, communication arms, and
request-zero HOF withholding. No new provider implementation or system prompt
is introduced. The cache/budget adapter scores final candidates. Exploratory
workspace simulations and RAG are disabled in this first release. Provider calls
are made only with explicit `--harness`.

Outward-enabled requests receive a frozen `CONTEXT/EXPLORATION_STATE.json` in
addition to existing HOF/SIL context files. Request zero and outward-disabled
arms do not receive it. Inward-disabled candidates remain outside training and
are merged only into the compatibility HOF at terminal aggregation.

## Archive and efficiency

- Every fitted design is stored in `exploration.sqlite`, with an exact cache key.
- Full evaluator diagnostics/trajectories are compressed in that cache, not
  duplicated in processing logs or live archive records.
- New cache entries separate `evaluation_summaries` from compressed `diagnostics`.
  Rate fitting and exact replay request summaries; retained fitted designs and
  Harness responses load diagnostics on demand. Older monolithic `evaluations`
  records remain readable without rewriting them. Cache keys remain exact.
- The default discovery archive retains up to **2000** compact topology elites.
  Region representatives are preferentially retained; excess duplicates are
  evicted first. If the number of regions exceeds capacity, the live archive
  cannot represent every region. Historical region-best summaries remain.
- Only the **top 30 retained elites** are eligible for structural imitation;
  at most **16** demonstrations are drawn per update, optionally balanced across
  regions within that top set. There is no full-archive policy replay.
- The conventional loss-ranked HOF has **30** entries and is maintained separately
  for viewer compatibility. It can differ from the diversity-preserving archive.
- Ranking is cached, eviction uses a lazy heap, and region descriptors use cheap
  reaction-role counts rather than repeated all-pairs graph distances.
- Parent groups and region weights are frozen once per cohort, preserving draw
  order. Related ledger writes share transactions; reservations still commit
  before evaluation under SQLite `synchronous=FULL`. Payload decoding is outside
  the shared database lock.

`reaction_roles_v1` groups designs by the number of output-producing,
output-consuming, catalyst-sharing, source, sink reactions and participating
species. Fixed template reactions are omitted. `topology` instead uses exact
labelled topology identity. Neither establishes mechanistic diversity or prunes
functionally redundant reactions; the descriptor must be reviewed before a
paper campaign. Historical successful-region count differs from live HOF size.

## Learning and numerical targets

Canonical physical loss is unchanged. Successful-region utility is
`Q * (1 + eta * clip(1 - loss / threshold, 0, 1))`, where Q requires a complete
simulation and the declared loss cutoff, optionally also all binary rows correct
with margin >1e-6. The episode reward is the nonnegative increase in historical
region-best utility plus `beta * exp(-max(loss,0)/threshold) / sqrt(1+visits)`.
Invalid/failed simulations receive no novelty. Visits count committed fitted
design attempts, including repeats; raw proposal attempts are logged separately.
The imitation term is mean structural negative log likelihood per reaction.

The first release uses a running reward baseline, not the old risk-seeking
loss-quantile update. Four archive/budget summary features condition the reused
encoder/head via a new context projection. It does not yet learn a requested
target-region policy. Fixed-policy cohorts prevent stale-policy gradients;
numerical evaluations run concurrently, but updates wait for their own cohort.
Model calls do not block epoch updates. There is no adaptive source allocation.

The structural policy retains the **10240-dimensional** intermediate representation.
Its rollout maintains batched observation tensors, without constructing or compiling
partial CRNs. Sampled prefix states are reused for gradient replay; imitation
prefixes are also constructed in tensors. The ordinary explicit observer and
tensorizer define the equivalence contract. Fixed-template parameters, action masks,
loss equations and entropy checks are unchanged. Other observation contracts are
rejected rather than silently approximated.

The exploration runner's session builder does not allocate unused rate-head
weights. A metadata-only head and bounded CPU scratch buffer consume the original
initialization RNG draws, preserving structural weights and seeded sampling. The
builder still returns the standard `Session` type. Supplying an existing legacy
Session to the trainer remains supported; its externally owned rate head is not
removed. Network-size ablations use explicit overrides and separate run directories;
they do not change the production default.

Rate fitting uses log coordinates in [0.001,100], unchanged fixed template rates,
and best-valid-so-far retention. Available backends are rate-only evolution
(single-coordinate Gaussian mutations and 20% random restarts) and bounded
Powell. RL/random seeds start at rate 1; archive mutations inherit unchanged
parent rates and initialize the replaced reaction at 1. This initialization
difference is explicit and must be controlled in source-comparison ablations.
Equivalent reaction orderings use the same canonical fitting order.

Optional `structural_refinement_steps` adds attributable single-parent edits
after fitting. Each edit gets its own bounded fit; total evaluations still share
the campaign cap. Credit returns to the starting RL proposal, but records identify
GA descendants and parent hashes. This is not a multi-parent genetic crossover.
CVODES sensitivities and neural ODE training are not implemented.

## Run

From the repository root, use the existing virtual environment:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m comparisons.search_exploration.run \
  --task logic \
  --exploration-config comparisons/search_exploration/configs/smoke.json \
  --output comparisons/search_exploration/runs/my-smoke
```

`configs/pilot.json` is a larger Logic pilot, not a launched campaign. For the
RPA/Powell infrastructure check use `--task rpa` with
`configs/smoke-powell-rpa.json`. Task configurations, quality criteria and reaction
budgets must be chosen explicitly for other tasks; Logic thresholds do not
automatically transfer. Deterministic CVODE mass-action tasks with one scalar
rate per added reaction are supported. SSA is explicitly rejected.

Add `--comet-project <approved-project>` to use existing Comet credentials.
The logger uploads parameters, metrics and figures; no keys are accepted on the
command line. Use an approved destination for live uploads.
The smoke upload to `genai-net-exploration` was explicitly approved and completed
on 2026-09-14. Automatic source, Git and machine-metadata logging is disabled.
Local logging/figures work without Comet. Add `--harness --dsh-home <configured-home>`
only when model calls are intended; `--model`, `--llm-provider`, `--llm-base-url`,
`--communication-mode`, `--llm-every` and `--task-prompt-variant` are available.

## Artifacts and compatibility

- `manifest.json`: settings, evaluator/dependency/source fingerprints and model mode.
- `results.sqlite`: existing ResultsDatabase schema, readable by the current viewer.
- `exploration.sqlite`: compressed cache, budgets, visits, processing states, designs, events.
- `checkpoint.pt`: policy AND optimizer, RNGs, archive, visits, pending cohort,
  committed epoch, and available Harness history/self-memory/provenance.
- `pending.pt`: a small pre-dispatch runtime journal referencing the durable model
  checkpoint by generation ID. It contains no policy/optimizer tensors. Full
  checkpoints are written initially and after updates, not before every cohort.
- `cohorts/`: compact proposal/outcome records, referring to cached diagnostics.
- `metrics.jsonl`, `status.json`, `completed.json` or `failed.json`.
- `progress.png`, `progress.pdf`, `sources.png`, and periodic HOF JPGs.

Existing trainer-style `step_epoch`, HOF access, results-database configuration,
Harness configuration, and save/load entry points are retained by the adapter.
Checkpoint formats are intentionally distinct; legacy checkpoints are not silently
migrated. `sample()` uses budgeted fitting under the current task. Explicit task
overrides retain the legacy diagnostic sampler, use constant initial rates and
are logged as outside-training evaluations rather than mixed into the cache.

The snapshot database preserves emitter values RL/GA/RANDOM/LLM and adds explicit
`*_seed_refined` classes for non-LLM proposals. The sidecar keeps origin, parent,
parameter operator and policy version. Old analysis scripts assuming only RL/LLM
are **not scientifically applicable without adaptation**, even though the SQL
schema remains readable. The legacy `cumulative_rl_evaluations` column contains
combined non-LLM refinement work in this new method; detailed source attempts and
global budget counters are in the sidecar/metrics. Do not compare that field with
legacy raw RL rollout counts without consulting the method manifest.

## Failure handling and resume

Evaluators are persistent spawned processes. Call deadlines terminate and replace
the affected worker. Programming/evaluator errors fail the cohort visibly; timeouts
and invalid scientific outcomes remain recorded. Evaluation reservations are
durable and conservatively charged even if a process dies before its result is
stored. Budget counters are not inferred from epochs, and caching does not make
duplicate proposals invisible.

For interrupted runs, repeat the identical command with `--resume`. Completed
cohort jobs are reused; pending work is reconstructed from its frozen proposals.
Keep `checkpoint.pt`, `pending.pt`, the SQLite store and cohort files together.
A matching pending journal overlays the base checkpoint's runtime state; an older
journal cannot replace a newer committed checkpoint. Both file contents and parent
directory entries are fsynced. An interrupted Harness dispatch intent fails closed,
even if the external request had not yet been registered in memory.
Different code, evaluator, config, or budget is rejected. Use
`--resume --retry-failed-cohort` only after diagnosing a fitting failure; failed
job artifacts are preserved under distinct names. Numerical timeouts can affect
outcomes, so exact restart equality is only asserted for deterministic completed
evaluations, not identical wall-clock schedules.

An interrupted external model request cannot be resumed from RAM; the runner
refuses automatic replay when its checkpoint records pending LLM work. Such runs
need explicit reconciliation. No SIGSTOP-based continuation is used. Checkpoints
are trusted local pickle artifacts and must not be loaded from untrusted sources.

Processed means a particular initialization/optimizer budget was spent, not that
the topology is exhausted. There is no permanent hard topology exclusion. A new
initialization/budget can be tried in an explicitly configured new run; failed fits
remain retryable. Physical loss caches and novelty/credit ledgers are separate.

## Preservation and verification

```bash
.venv/bin/python -m comparisons.search_exploration.freeze_legacy --verify
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m pytest -q \
  tests/test_search_exploration.py tests/test_search_exploration_integration.py
```

The baseline archive preserves 214 selected scientific files as they existed in
the dirty worktree before this implementation. New sources, tests, outputs and
documents are confined to the new namespace. See `IMPLEMENTATION_LOG.md` for
changes, verification outcomes and remaining limitations.

For a reproducible CPU policy/cache timing check (no CVODE or LLM calls):

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m comparisons.search_exploration.benchmark_optimizations \
  --output comparisons/search_exploration/runs/my-optimization-benchmark
```

This produces `benchmark.json` and `policy_timing.png/pdf`. Timing ratios are
microbenchmarks, not end-to-end campaign speedups. Full trajectory storage still
grows with the number of physical evaluations; summary caching does not discard
those records or solve the long-run disk-capacity requirement.

To compare intermediate dimensions without changing the production model:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m comparisons.search_exploration.benchmark_nn_sizes \
  --output /local1/rossin/search-exploration/my-nn-size-benchmark \
  --embeddings 1024 10240 --tasks logic rpa --samples 16 128 \
  --elites 16 --repeats 5 --warmup 2
```

Each task/dimension/batch runs in its own CPU process. Outputs include a source
manifest, raw timings, parameter and memory counts, matched-setting checks,
JSON/CSV summaries, a Markdown report, and PNG/PDF timing and memory plots.
The imitation bank is identical between sizes for each task. Fixed synthetic
rewards exercise the optimizer; no evaluator or LLM is called. This compares
computational cost, not search quality or convergence. The 10240 default remains
unchanged. Use a new output directory for each comparison.

## NN search-quality pilot

`nn_size_quality.py` runs the actual CVODE-backed exploration method at both
intermediate dimensions. The 2026-09-14 pilot is under
`/local1/rossin/search-exploration/nn-size-quality-20260914`, with five matched seeds
for Logic and RPA (20 runs total), 20 epochs, 32 proposals per epoch, 16 fitting
calls per proposal and a 10240 physical-evaluation cap. Each task/seed pair shares
one configuration file; only the explicit `--embedding-size` and output differ.
No Harness calls are enabled. Existing size defaults are not changed.

The pilot logs to Comet project `genai-net-exploration`. `progress.json` records
queue/running/failure status, `logs/` contains per-run stdout, and `runs/` contains
the usual snapshots, exact evaluation ledger and checkpoints. A `source.tar.gz`
and per-file hashes preserve the implementation. Source changes stop further
dispatch rather than silently mixing implementations. Failures hold queued jobs
while active jobs finish; no automatic retries are performed.

Run this to refresh the quality report and plots from saved data only:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MPLCONFIGDIR=/local0/tmp/mpl \
  .venv/bin/python -m comparisons.search_exploration.analyze_nn_size_quality \
  /local1/rossin/search-exploration/nn-size-quality-20260914
```

The controller also generates this analysis when the queue finishes or holds.
Endpoint comparisons use complete matched seed pairs only and report undefined
losses separately. Actual evaluation counts accompany the shared budget cap:
caching and timeouts can affect consumption. These five seeds are exploratory
quality evidence, not a statistical equivalence test. Creating a `HOLD` file in
the campaign directory prevents further dispatch without stopping active jobs.
