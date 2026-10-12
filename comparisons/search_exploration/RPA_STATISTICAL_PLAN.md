# RPA Workflow Comparison: Frozen 20-Seed Protocol

## Question and Scope

Does the Julia AdaGrad hybrid discover more useful RPA topologies, or achieve
better loss, than the evolution-fit hybrid and bounded standalone GA?
This is a comparison of complete recipes, not a pure optimizer swap or proof
that RL itself causes an advantage. The seed-0 runs informed these recipes and
are development evidence only. Existing results remain untouched.

## Design

- Twenty matched seed identifiers, 1 through 20, in all three arms: 60 new runs.
- Canonical RPA CVODE task from the corrected-stream reference manifest; nine
  input scenarios, 1000 time points through t=100, rtol=atol=1e-8. Five added
  distinct reactions, existing forbidden-reaction mask, rates [0.001,100].
- Frozen qualification: complete simulation and loss < 0.016491168958935352.
- Hybrids: 100 epochs, batch 1023, 1024-dimensional NN intermediate layer,
  50% RL / 25% mutation / 25% random proposals, existing learning recipe.
- Discovery archive 2000, compatibility HOF and top imitation pool 30,
  imitation sample size 16. Standalone GA retains its original 30-parent archive.
- Julia fitting: persistent AdaGrad workers; inherited/perturbed/two log-uniform
  starts, 24-attempt fitting budget, at most eight 24-attempt extensions per
  epoch, 20% measured fitting-worker-time target; frozen JuliaRecipe JSON.
- Evolution fitting: original 16-attempt rate fitting, 256 admissions per epoch,
  queue capacity 512. Julia admissions/queue capacity remain 32.
- No LLM, no rate-pruning penalty, no post-search deletion/pruning, no GPU.

## Budgets and Execution

All arms have a **214793-attempt ceiling**. Each hybrid reserves 102300 slots
for its scheduled direct search, leaving at most 112493 optimizer slots.
This is a new common-ceiling protocol: the old evolution-fit run had a larger
configured ceiling and used 133906 optimization evaluations; it is not reused
as a matched replicate. Hybrids are not forced to spend unused credit. GA stops
at 214793 actual evaluations; its batches of 1023 are not RL epochs.

One canonical evaluation and one joint loss/gradient evaluation each consume
one accounting slot but **are not equal compute**. Record canonical and joint
counts, actual total, runtime, failures and asynchronous fitting diagnostics.
Endpoint comparisons are at the stated stopping rules, not at equal CPU time.
Runtime is descriptive under concurrent load. Julia gradients and fitting
admission/initialization policies also differ from the evolution-fit arm.

One hybrid lane (32 search + 16 fitting/validation workers) runs alongside one
16-worker GA lane: at most 64 evaluator/fitter workers, plus lightweight
controllers and training. Julia splits its 16 fitting/validation workers into
eight persistent Julia fitters and eight canonical validators. BLAS/OpenMP
threads are capped at one. Hybrid order alternates by seed. Shared seed labels
do not imply identical proposal banks or wall-clock completion schedules.

All metrics/figures are local. Comet support remains available in individual
runners, but this campaign does not upload anything. Allow roughly 180 GiB of
artifacts based on the development runs; require 250 GiB free before launch and
hold new dispatch below 50 GiB. No old data are deleted.

## Endpoints and Statistical Analysis

The experimental unit is one run/seed, not one candidate or one epoch.

Primary endpoint: count of distinct qualifying **discovered** topology hashes
in the complete design ledger, including designs evicted from the live archive.
This avoids comparing archive capacities of 30 and 2000. It measures structural
solution yield, not necessarily mechanistic diversity or irredundant designs.
Secondary endpoint: rank-one canonical loss at the completed search boundary.

Two planned contrasts: Julia vs evolution-fit, Julia vs standalone GA. For each,
report paired W/T/L, mean paired difference and its 95% paired-seed percentile
bootstrap interval, and median paired ratio with its interval where denominators
are positive. Bootstrap: 10000 resamples, fixed analysis seed 20260915.

Use two-sided paired Wilcoxon signed-rank tests with Pratt handling of zero
differences and exhaustive sign permutations (at most 2^20 assignments).
For loss tests/counts, numerical ties use rtol=1e-9, atol=1e-12; integer yield
ties are exact. All-zero differences give p=1. Apply Holm correction to the
four-test family, alpha=0.05. Wilcoxon interpretation assumes symmetric paired
differences; uncertainty is conditional on this task and frozen recipe. Bootstrap
intervals are marginal, not multiplicity-adjusted. No equivalence conclusion
follows from non-significance, and 20 seeds do not guarantee adequate power.

No interim p-values or data-dependent stopping. Partial figures are explicitly
descriptive. Formal inference requires all 60 planned runs to finish and pass
identity/budget checks. A failure holds new dispatch while active runs drain;
it remains in the record, without automatic retries, replacement seeds or silent
complete-case exclusion. Any recovery requires diagnosis and a documented
protocol amendment before restarting. Source/config changes also hold dispatch.

## Reproduction and Monitoring

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m comparisons.search_exploration.rpa_statistical_campaign \
  --output comparisons/search_exploration/reports/rpa-statistical-20seed-20260915 \
  --project /local0/rossin/git/SynthEvo/SynthEvo --prepare-only

tmux new-session -d -s rpa-statistical-20seed-20260915 \
  'cd /local0/rossin/git/GenAI-Net && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m comparisons.search_exploration.rpa_statistical_campaign --output comparisons/search_exploration/reports/rpa-statistical-20seed-20260915 --execute-prepared'
```

`plan.json`, `plan.sha256`, copied configs and `source.tar.gz` freeze the run.
`progress.json` updates every 15 seconds; `logs/` has stdout per run. Reports,
PNG/PDF endpoint plots and CSV/JSON summaries are refreshed every five minutes
and when the queue ends. Each run also retains its existing time-course plots,
checkpoints and viewer-compatible SQLite database. Inspect `analysis/REPORT.md`.
Creating a `HOLD` file in the campaign folder prevents new dispatch without
interrupting active work. There is no automatic restart of a started controller.

Read-only refresh:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m comparisons.search_exploration.analyze_rpa_statistical \
  comparisons/search_exploration/reports/rpa-statistical-20seed-20260915
```

Statistical API references:
[SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html),
[SciPy paired permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
