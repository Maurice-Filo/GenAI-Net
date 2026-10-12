# RPA Four-Arm Engineering Comparison

## Design

Single seed (0); canonical RPA/CVODE task, five admissible distinct added
reactions, shared [0.001, 100] rate bounds, no LLM or replay. Existing completed
hybrid and no-RL runs are reused without modification.

| Arm | Structural proposal mix | Rate treatment | Parent/training archive |
| --- | --- | --- | --- |
| Hybrid v2 | 50% RL, 25% archive mutation, 25% random | Selected independent 16-call rate fits | Corrected 2000 entries, imitation from top 30 |
| No RL | 25% archive mutation, 75% random | Same selected independent fits | Same 2000 entries; no policy/imitation updates |
| No archive mutation | 50% RL, 50% random | Same selected independent fits | Same 2000 entries and imitation |
| Standalone GA, bounded v1 | Original 45% rate mutation, 45% reaction replacement, 10% restart | One evaluated offspring per GA proposal; no separate fitting stream | Original top-30 lowest-loss topology archive |

The no-mutation arm tests removing archive structural mutation, not removing the
evolutionary rate fitter. It is not the historical joint structure/rate RL-only
algorithm. Random replacement keeps total structural proposal counts constant.

The standalone algorithm imports the original `Evolution` and `EliteArchive`
from `comparisons/llm_seeded_evolution/search.py`. A new `BoundedEvolution`
wrapper only clips proposed rates. It preserves the old log-uniform [0.1, 50]
initialization prior, mutation probabilities and scale, uniform top-30 parent
selection, and batch-frozen parents. The original GA source is unchanged.
There is no crossover, no pretrained policy, and no saved LLM proposal replay.
The original mutation operator can exceed the new bounds; therefore this is a
separately named bounded comparison, not a relabeling of the older experiment.

## Budgets And Fairness

The three stream arms use 100 epochs x 1023 structural proposals, with the same
1636800-evaluation cap, 32 search / 16 optimization workers, queue 512 and up to
256 fits scheduled per epoch. Queue timing changes actual fit admissions and
consumption. The full hybrid consumed 214793 actual evaluations; no RL consumed
211477. This is not an exact fixed-evaluation ablation between those arms.

Standalone GA targets exactly **214793 physical candidate evaluations**, using
1023-offspring batches and a shortened final batch. Exact duplicate rate/topology
vectors use the same cache machinery; cached proposals do not spend this budget.
Proposal attempts are separately logged. A visible failure stops the run if a
safety cap of ten times the physical budget is reached before spending the
budget; no silent successful termination. One physical evaluation scores the
entire canonical candidate over all task scenarios, not a single ODE trajectory.

Do not equate GA batches with RL epochs. Compare curves against evaluations and
elapsed time, plus explicitly recorded checkpoints below the shared cost limit.
Different initialization priors and parent archives remain algorithm choices.
Wall times are affected by concurrent load; these are single-seed engineering
results, not significance tests or a complete factorial interaction experiment.

## Diversity Accounting

Primary comparison: all distinct qualifying topology hashes discovered in the
complete design ledger, including evicted entries. First qualification is
assigned to its recorded search/merge boundary. The metric is not capped at
30 for GA or 2000 for the stream methods. Retained archive counts remain in
the JSON summary for transparency. All-discovered reporting removes the archive
size reporting cap, not the effect of parent archive size on the search itself.

## Execution And Artifacts

Campaign root:
`/local1/rossin/search-exploration/rpa-remaining-controls-20260915`.

Tmux: `rpa-remaining-controls-20260915`. Both new jobs run concurrently, with
48 evaluator workers for no-mutation plus 16 for standalone GA. CPU only,
normal priority, no system/GPU configuration changes. Failed outcomes remain
recorded; no automatic retry or replacement. Previously completed runs are not
relaunched. On success, the controller audits completed artifacts and generates
four-arm JSON, Markdown, and PNG/PDF comparisons under `analysis/`.

```bash
env -u COMET_API_KEY CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/local0/tmp/mpl \
  .venv/bin/python -m comparisons.search_exploration.run_remaining_controls \
  --reference /local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/run \
  --no-rl /local1/rossin/search-exploration/rpa-streams-no-rl-100epoch-20260915/run \
  --output /local1/rossin/search-exploration/rpa-remaining-controls-20260915
```

The command refuses an existing campaign output directory. Consult
`progress.json`, `no-mutation/run/status.json` and `standalone-ga/run/status.json`.
Each `run/completed.json` indicates its algorithm has completed; campaign
`completed.json` also confirms the final combined analysis. It must not be
inferred from a tmux session merely disappearing.

Comet experiments:
- No mutation: https://www.comet.com/redsnic/genai-net-exploration/5a3863b6d4f3445d80af7572d569d7d4
- Standalone GA: https://www.comet.com/redsnic/genai-net-exploration/7e7c0842b79546a3b000d19782768f64

The generic per-run GA archive plot reflects its actual 30-slot parent archive;
the combined `--discovered` comparison reports all discoveries. The generic
archive-utility source-gain plot is not a GA attribution analysis: standalone
GA does not use the stream method's descriptor-utility objective.

## Verification

62 focused tests passed, including exact operator equivalence without clipping,
bound clipping with unchanged topology/parent, distinct plot labels, no-RL
update exclusion, no-mutation proposal exclusion, and discovery counting across
evictions and repeated topology sightings. All 214 frozen legacy
source checksums verified; no original evolutionary source was edited.

Canonical GA smoke at
`/local1/rossin/search-exploration/rpa-standalone-ga-smoke-20260915/run` completed
64 physical evaluations, 43 mutation and 21 random proposals, with no runtime
failures. Eight incomplete simulations are explicit scalar outcomes, not
successful trajectories. Audit passed bounds, structure, source hashes and
budget reconciliation. This is software verification, not a performance arm.
`reports/controls-analysis-smoke-20260915` is likewise a plotting/software smoke,
not the final four-arm comparison.
