# Independent Search And Rate Optimization

Current implementation: **search-streams-v2**, with corrected archive utility
ordering. The completed v1 trial below is retained as diagnostic evidence, not
silently relabeled as the corrected implementation.

## Scope

Added `RL4CRN/search_streams/` and separate runner/analysis commands. Existing
`RL4CRN/search_exploration/`, legacy scientific code, settings, prompt files and
old outputs are not modified. The running synchronous RPA reference is preserved.
The new command is CPU-only and enables no LLM or RAG calls.

## Operational Workflow

1. At a search boundary, merge the currently completed optimization results in
   submission order. Freeze the archive/context used to sample the next cohort.
2. Sample RL/archive-mutation/random structures with the existing probabilities
   .5/.25/.25. As before, archive mutation inherits surviving parent rates;
   RL/random structures start with rates 1. Empty-archive mutation falls back to
   random. These sources are not a controlled comparison of initialization schemes.
3. Evaluate each search proposal once using canonical CVODE. Only immediate
   results provide the on-policy reward. Advance RL without waiting for optimizer
   jobs; already available elites supply structural imitation.
4. Admit up to 256 fitting jobs per full-run epoch, subject to a 512-job capacity
   including active jobs. Roughly half are selected by initial loss, a quarter by
   low descriptor-region visitation, and the remainder randomly. Duplicate exact
   initial reaction/rate vectors are not queued again during the same run.
5. Independent optimization workers execute the existing seeded log-rate
   evolutionary fitter. The initial loss is reused from the exact cache. Each job
   has the existing 16-call/60-second fitting limit and 30-second evaluation limit.
   A valid structure with incomplete initial trajectories remains eligible for
   coverage/random fitting slots because different rates can rescue its behavior.
6. Improved designs return to the archive at later search boundaries. They can
   become mutation parents and elite-imitation demonstrations. Their delayed loss
   is not assigned as an ordinary on-policy reward to a changed RL policy.
7. After the requested search epochs, drain every dispatched optimizer job,
   merge its outcome, write the terminal HOF, checkpoint, metrics and figures.
   Only then write `completed.json`.

The search stream still waits for its own direct-evaluation cohort, preserving
the on-policy update contract. It does not wait for the optimization stream.
Completion timing affects which elites are visible at each boundary; this is
not bitwise deterministic across different machine loads. Dispatch, merge epoch,
policy version, parent, source, origin and parameter operator are recorded.

## Cache, Storage And Budgets

- Exact keys pair reaction IDs with `float.hex` rates in canonical order and
  include evaluator identity. No rate rounding or lossy key equivalence.
- Bounded in-memory scalar lookup backed by SQLite WAL with FULL durability.
- Every evaluation has a durable reservation and scalar outcome/failure record.
  Duplicate simultaneous keys coalesce rather than spending the budget twice.
- Each optimizer worker keeps its intermediate trajectories locally. Only the
  best result's compressed diagnostics are transferred after a fitting job.
- Search workers transfer compressed diagnostics once per direct evaluation.
  Rich diagnostics are retained for archive admissions; historical admissions
  remain on disk. This is selective storage, not a strictly fixed-size database.
- A cached scalar result can legitimately lack trajectories. It remains usable
  for scoring. We do not silently recompute diagnostics or invent missing plots.
- Existing scalar evaluator consumers have a compatibility adapter; it accounts
  once and retains their requested diagnostics. The fast streams bypass this
  full-payload adapter. No external model was called to test the adapter.
- Reserve 102,300 physical evaluations for direct search (100 x 1023). The
  optimizer cannot consume that allocation. Both streams share the 1,636,800
  total cap; separately log actual search and optimization consumption.
  One evaluation means a complete canonical candidate score over the task's
  scenarios, not a single ODE trajectory. Pending reservations are charged
  conservatively until their outcomes are recorded.
- At most 25,600 jobs are admitted under the configured per-epoch limit, and
  typically fewer when the queue is full. Therefore actual consumption can be
  well below the shared cap. This is a selected-refinement algorithm, not simply
  a faster implementation of refining every candidate.
- No automatic failed-job retries or failure-based exclusion from analysis.
  Errors and timeouts remain visible. Terminal draining can extend runtime.

## Completed Verification

- 54 tests passed across the stream tests and four existing exploration suites
  after the final compatibility fix (19.68 s).
- Exact key ordering/rate sensitivity, in-flight coalescing, protected budgets,
  charged failures, selective diagnostics and cache reopening are covered.
- A blocked optimizer test completed two RL epochs without optimizer progress;
  draining later improved the archive without another policy update.
- Provenance tests distinguish direct search from rate-refined descendants.
- Real CVODE worker fitting reproduced the reference loss and fitted rates
  exactly, reused the starting loss, and tested single-charge compatibility calls.
- All 214 frozen legacy scientific files passed checksum verification.

Matched fixed-workload benchmark (32 frozen RPA proposals, 16 calls each, eight
workers; startup excluded):

| Path | Evaluations | Time | SQLite size | Fitting failures |
| --- | ---: | ---: | ---: | ---: |
| Original full-transfer | 512 | 19.154 s | 131,665,920 bytes | 0 |
| Compact worker-local fitting | 512 | 10.794 s | 487,424 bytes | 0 |

All best records and objective-call counts matched exactly. Measured throughput
was 1.77x higher for this workload. This is one measurement with the old path
first, not a randomized repeated performance study. The compact benchmark retains
scalar records only: an actual trainer additionally saves selected archive
diagnostics. Do not extrapolate the table into a 270x campaign-size reduction.

Artifacts: `/local1/rossin/search-exploration/stream-worker-benchmark-20260914`.

Completed search-quality validations (seed 0, 20 epochs, batch 32):

| Run | Actual evaluations | Time | Best loss | Qualifying topologies |
| --- | ---: | ---: | ---: | ---: |
| Existing synchronous pilot | 10200 | 747.45 s | 0.15750348 | 0 |
| Initial asynchronous validation | 2335 | 126.81 s | 0.15266600 | 0 |
| Asynchronous with incomplete-start eligibility | 2271 | 132.21 s | 0.15459799 | 0 |

Both asynchronous runs completed all epochs and drained their queues with no
fitting failures. The latter merged 109 jobs, 106 of which improved their starting
loss. The two asynchronous rows are successive implementation validations, not
independent replicates or a controlled test of the eligibility safeguard.
The later scalar-evaluator compatibility adapter was unit/CVODE tested separately;
its source identity is included in the subsequent full-run manifest.

These are one-seed exploratory results with different consumed budgets and
runtime environments. They do not establish statistical superiority, equality
of search quality, or successful RPA designs. The actual-evaluation plots should
be consulted rather than interpreting endpoint wall-time ratios as pure speedup.

## Full Run And Monitoring

Controller tmux: `rpa-streams-evaluation-20260914`.
Root: `/local1/rossin/search-exploration/rpa-streams-100epoch-20260914`.

- `plan.json`: exact child command and reference path.
- `progress.json`: current statuses of both runs, refreshed every 30 seconds
  while asynchronous search runs, every minute while waiting for the reference.
- `run/manifest.json`: frozen scientific configuration and source hashes.
- `run/source.tar.gz`: executed-source snapshot, including the legacy snapshot.
- `run/status.json`, `run/metrics.jsonl`: epoch, budget, loss, qualifying counts,
  pending/merged optimizer jobs, timing breakdown, source gains and policy metrics.
  The per-epoch `Source/*/gain` metrics describe immediate search contributions
  only; delayed fitting contributions are in the archive event ledger and the
  analysis JSON's `stream_contributions`. Do not interpret `sources.png` as the
  complete attribution of both streams.
- `run/comet.json`: experiment identifier in `genai-net-exploration`.
- `run/results.sqlite`: viewer-compatible HOF and admission records.
- `run/exploration.sqlite`: complete scalar cache, budget and event ledger.
- `run/optimization_results.json`: individual fitting improvements and merge epochs.
- `run/completed.json`: search complete and all optimization jobs drained.
- `analysis/REPORT.md`, JSON and PNG/PDF plots: generated automatically after
  asynchronous completion and refreshed after the synchronous reference finishes.
- `controller_failed.json`/`run/failed.json`: explicit failures, never auto-retried.

The 100-epoch run uses batch 1023, NN intermediate size 1024, 32 search workers,
16 fitting workers, original bounds/threshold/reward/imitation settings, and Comet
metrics/plots. It runs independently of the existing synchronous reference.

### Completed Full-Run Outcome

This subsection records the completed **v1** run. A later audit identified the
prototype archive-retention defect described below; its measured losses and
counts remain valid for the executed code.

The asynchronous run completed all 100 epochs and drained all 8666 dispatched
optimization jobs. Total elapsed time, including initialization, was 1397.58 s
(23.29 min). Best canonical loss: **0.002742564523021603**; qualifying archived
topologies: **57**. The physical evaluation total was **206870**, split into
77235 direct-search and 129635 optimization evaluations. There were 34068 cache
hits and 18 coalesced in-flight lookups. Of the merged optimization jobs, 8475
improved their starting loss.

Final audit: all charges reconcile, source snapshot hashes and evaluator identity
match, no failed evaluation events or malformed recorded reaction counts, no
out-of-bound recorded rates, and zero pending optimization jobs. The scalar
ledger explicitly retains 15117 incomplete simulations. These are not counted
as successful trajectories. Rich diagnostics were retained for 15071 evaluations.

At the comparison snapshot the synchronous reference was still running (22
epochs, 352241 evaluations, loss 0.00571082, four qualifying topologies). Under
the shared 206870-evaluation limit, its last recorded boundary was 195603
evaluations, loss 0.05095534 and zero qualifying topologies. These are distinct
recording boundaries, not an exact matched-budget endpoint test. This is promising
single-seed evidence for the new pipeline, not statistical proof of superiority.
The controller will refresh the comparison after the synchronous reference ends;
the reference was not stopped or restarted.

Completed experiment:
https://www.comet.com/redsnic/genai-net-exploration/704919d5c66f48689a25cdec7ca4e88f

### Archive Correction And V2 Evaluation

The inherited prototype's eviction heap ordered non-representatives by negative
utility, so a higher-utility candidate could be evicted before a poorer one.
A two-slot regression retained losses 0.001 and 1.0 while discarding 0.002.
`search_streams/archive.py` corrects the sign for the new mode, retaining the
existing region-representative protection and using lower utility as the first
eviction choice within that priority class. The frozen synchronous prototype
and all earlier scientific files remain unchanged.

An offline replay of the complete v1 discovery sequence reproduced its original
terminal archive membership exactly. Applying the corrected rule to those same
observed records retained all 137 qualifying topologies instead of 57, without
any evaluator or model calls. This isolates the retention defect; it does not
predict the learning trajectory of a corrected rerun. Replay JSON and PNG/PDF:
`/local1/rossin/search-exploration/stream-archive-retention-replay-20260914`.

This affects portfolio retention and potentially subsequent learning. The v1
ledger contains 137 discovered qualifying topologies while its final live
archive held 57. We do not rewrite its logged counts. V2 is an additional
algorithmic difference from the synchronous reference, not an effect that can
be attributed to asynchronous scheduling alone.

The corrected full campaign is isolated at
`/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914`, with tmux
`rpa-streams-v2-evaluation-20260914`. It uses the same 100-epoch schedule, seed,
batch, worker split and budgets. Comet:
https://www.comet.com/redsnic/genai-net-exploration/3946a0d47774489dbb434b9c122e9818

Verification including the new eviction regression: 54 tests passed (201.59 s
under concurrent workload).

### Corrected V2 Completed Outcome

The corrected campaign completed all 100 epochs and drained all 8951 optimization
jobs in 1685.01 s (28.08 min), including initialization. Best canonical loss was
**0.0025487682658084914**, with **176 qualifying topologies** retained in the
2000-entry archive, spanning 55 qualifying descriptor regions. Descriptor regions
are not a mechanistic-diversity measure. Of the fitting jobs, 8726 improved their
starting loss. There were 214793 physical evaluations (80887 direct search and
133906 optimization), 30707 cache hits and 16 coalesced in-flight lookups.

The final audit verified frozen source hashes, evaluator identity, charge and
dispatch/merge reconciliation, recorded structures and recorded rate bounds.
There were no runtime failure events or pending jobs. The ledger explicitly
contains 15203 incomplete simulations; 15235 evaluations retain rich diagnostics.
No LLM was used.

The synchronous reference remained active at the report snapshot: 26 epochs,
413141 evaluations, loss 0.005443852218968288, and 11 retained qualifying topologies.
At its last boundary below the shared 214793-evaluation limit it had consumed
211865 evaluations, with loss 0.0509553354751609 and zero qualifying topologies.
These are not exactly paired evaluation boundaries. The archive correction,
selective fitting and changed on-policy rate treatment preclude attributing the
gain solely to asynchrony. This is one-seed engineering evidence, not a statistical
comparison. The reference continues unchanged; the controller refreshes the report
after it finishes. The comparison PNG was visually inspected.

Report and PNG/PDF figures:
`/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/analysis/`.

After completion, the optional diagnostic cache adapter was made to return an
explicit empty `task_info` dictionary when diagnostics were not retained. This
preserves compatibility with inherited diagnostic sampling without recomputation.
Scalar worker lookups and training behavior are unchanged; the frozen source tar
remains the authority for the code actually executed in this completed run.

## Limitations And Audit

- This is a one-seed algorithm trial, not a manuscript-ready comparison cohort.
  Actual consumed evaluations and elapsed time must both be reported.
- The delayed optimizer changes the effective on-policy reward (starting-rate
  evaluation rather than fitted loss). This can change learning behavior and is
  not attributed solely to asynchronous execution or faster storage.
- Fixed quotas/queue capacity are engineering defaults, not optimized settings.
  Low starting loss is not a reliable proxy for a topology's best possible loss.
- Descriptor coverage remains a role-count proxy, not mechanistic diversity.
- Checkpoints preserve policy, RNG, archive and dispatch metadata, but automatic
  asynchronous resume is explicitly refused pending job reconciliation. No claim
  of deterministic recovery from arbitrary power loss is made.
- The new CLI exposes no LLM launch flags. Existing Harness/provider code and
  prompts are preserved; no paid end-to-end LLM test was performed.
- During implementation `/local0` filled. Completed smoke artifacts (2.6 GiB)
  were relocated intact to `/local1/rossin/search-exploration/completed-smokes-20260914`;
  the original `comparisons/search_exploration/runs` path is a symlink. No experiment
  or model files were deleted, and the active reference outputs were untouched.
