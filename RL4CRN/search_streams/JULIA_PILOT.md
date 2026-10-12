# Opt-in Julia Fitting and Separate Pruning

The previous evolution/Powell runner defaults remain unchanged. Enable the new
path with `--julia-project /path/to/SynthEvo/SynthEvo`; this uses AdaGrad,
irrespective of the old `ExplorationConfig.optimizer` field. The effective
operator is recorded separately as `julia_adagrad`. No LLM is used by this runner.

## Search Recipe

- At most 32 distinct topologies per epoch: target 50% promising feasible losses,
  25% least previously fitted topologies, 25% random; unused slots are filled.
- Eight persistent one-thread Julia workers, at most 32 active plus queued jobs.
- Four starts: inherited, one log-normal perturbation (sigma 0.5), two log-uniform
  starts over the rate bounds. Every rate is bounded to `[0.001, 100]` by default.
- Screen all starts, continue the two feasible starts with lowest loss. A total
  of 24 joint loss/gradient attempts covers screening, updates, and rejected
  backtracking trials. Split remaining credit between those two starts.
- Up to eight improving or previously unfitted topologies per submission epoch
  can receive another 24 attempts. These are the first eligible asynchronous
  completions, not a retrospectively ranked best eight. The same optimizer
  checkpoints continue without resetting AdaGrad accumulators or reevaluating
  the saved current gradient.
- Canonical CVODE independently verifies the retained best. Those extra calls
  are separately counted inside the global search budget. Only canonically
  complete results enter the archive. Differences beyond `abs=2e-6, rel=2e-4`
  are logged as warnings, with the canonical score used for all archive decisions.
  Only differences exceeding `max(1e-5, 0.01 * max(losses))` abort as substantial
  mismatches. Nonfinite/negative losses remain errors; canonical incompleteness
  rejects a candidate. Both scores and the policy are saved, including on failure.
- Direct-search budget is reserved. Julia jobs reserve credit atomically and
  refund only unused credit after a successful response. A killed worker has
  unknown cost, conservatively charged and recorded, and aborts the run.
- A soft admission controller targets 20% fitting worker-time. It uses measured
  search evaluator seconds, completed fitting worker elapsed seconds, and a
  pending-job cost estimate. This is not CPU profiling or a guaranteed partition;
  short runs and cold compilation can overshoot. Admissions adapt to measured
  costs; observed gains are used for extension eligibility, not a learned scheduler.
  Only one calibration job is admitted until its measured cost is available.
- Failed initial-rate evaluations do not blacklist a topology. Later visits use
  a new deterministic restart seed. Only one active job per topology is allowed.
- Exact complete screening checkpoints are cached (256 per worker); model
  preparation is cached (32 per worker). Canonical exact evaluations share the
  existing store. Internal trial evaluations are charged even if repeated.
- Ready results merge at epoch boundaries; only terminal draining waits. Original
  on-policy rewards are not retroactively replaced by fitted losses. Fitted
  elites affect future sampling and imitation through the existing archive.

`JuliaRecipe` fields can be overridden using `--julia-recipe settings.json`.
The supported adapters are deterministic single-output tracking/RPA and Logic
with absolute trajectory L1. Inputs, time grid, weights, initial conditions,
targets, tolerances, and concentration ceiling come from the executable task.
Unsupported objectives fail explicitly; this does not add SSA support.

## Separate Pruning

`--post-prune` takes up to five qualified, distinct terminal archive topologies.
Each parent is reverified. In stored reaction order, try removing one added
reaction and fit the remaining rates with at most 11 joint evaluations plus
one canonical check (12 total). Stop after one accepted deletion per parent,
or 60 total attempts including parent verification. Failed fits consume budget.

Acceptance requires canonical completion and

```text
child_loss <= min(quality_threshold,
                  parent_loss + max(1e-5, 0.02 * parent_loss))
```

Logic also requires all truth-table rows remain correct. Fixed starter/template
reactions cannot be selected for deletion. No rate sparsity penalty, rate zeroing,
or automatic small-rate pruning is used. Deleting the last free reaction is not
attempted by this pilot. Outputs, parent identities, removed reaction IDs,
checkpoints, rejected trials, and costs live under `post_pruning/`. The search
archive/database is not rewritten and these results are not search endpoints.

## Run a Small Pilot

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m comparisons.search_exploration.run_streams \
  --task rpa --task-config comparisons/rpa_search/configs/rpa_cvode.json \
  --exploration-config comparisons/search_exploration/configs/julia_stream_smoke.json \
  --output comparisons/search_exploration/reports/NEW-UNUSED-DIRECTORY \
  --search-workers 8 \
  --julia-project /local0/rossin/git/SynthEvo/SynthEvo --post-prune

.venv/bin/python -m comparisons.search_exploration.report_julia_stream \
  comparisons/search_exploration/reports/NEW-UNUSED-DIRECTORY
```

The smoke config is three epochs of 128 proposals, CPU-only, with no LLM calls.
`julia_stream_pilot.json` instead uses three epochs of 1023 proposals. The pilot is software
verification, not a paper comparison. Existing metrics, database snapshots,
plots and optional `--comet-project` integration are retained. External upload
is not performed by default. Do not upload without the applicable authorization.

Automatic campaign resume is deliberately unsupported: pending jobs need explicit
reconciliation. Per-fit optimizer checkpoints are saved and resumable only with
the identical model/configuration/backend fingerprints. `evaluation_limit` is a
runtime per-call cap, independent of those checkpoint settings; `max_evaluations`
keeps its earlier checkpoint-compatible semantics.
