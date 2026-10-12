# Exploration implementation log

## 2026-09-14: Scope and preservation

- Add an opt-in implementation under `RL4CRN/search_exploration/` and a separate runner.
- Preserve all existing scientific source files, defaults, prompts and campaign outputs.
- Freeze working-tree source bytes, not merely Git HEAD, before implementation.
- Keep canonical loss separate from discovery reward and imitation weights.
- Use an on-disk evaluated-design store, a configurable larger discovery archive,
  and a small top-ranked imitation subset. Preserve the existing viewer database format.
- Retain Comet metrics and figures; no credentials in source, manifests or commands.
- No long experiments or paid model calls are authorized by this implementation step.

## Planned milestones

1. Exact evaluation caching, atomic budget accounting, retryable processing states.
2. Bounded rate fitting with process-isolated CVODE and explicit failures.
3. Large discovery archive, visit counts, small elite imitation selection.
4. Structural policy updates and mixed RL/random/archive proposal sources.
5. Harness adapter, compatible database output, provenance, Comet and plots.
6. Durable checkpoints, focused tests, real-CVODE smoke tests and final audit.

Descriptor defaults and optimizer settings are engineering pilot defaults, not a
frozen scientific protocol or evidence of mechanistic diversity.

## Implemented

All additions are opt-in, in `RL4CRN/search_exploration/`, this directory, and
`tests/test_search_exploration*.py`. Existing scientific files were not edited.

- `freeze_legacy.py` and `baseline/`: checksummed archive of 214 scientific files
  from the actual working tree, including its pre-existing changes.
- `config.py`, `run.py`, and three JSON configurations: separate entry point,
  explicit task/configuration identity, exclusive run lock, CPU/device selection,
  bounded resource settings, opt-in Comet and Harness. Existing campaigns are
  neither overwritten nor automatically restarted.
- `store.py`: SQLite WAL cache, exact order-independent reaction/rate keys,
  evaluator fingerprints, duplicate-call coalescing, atomic evaluation reservations,
  processing journal, visit counts, and compact design records. Failures remain
  recorded and charged; they do not become successful cached evaluations.
- `evaluation.py`: persistent process-isolated canonical CVODE evaluators;
  null/fixed-reaction masking, independent candidate validation, finite bounded
  rates, and worker replacement after timeouts. No canonical loss was changed.
- `refinement.py`: log-rate evolution and bounded Powell, best-valid-so-far
  retention, explicit per-fit and global budgets, canonical fitting order,
  retryable failures, and exact replay without another numerical evaluation.
- `archive.py`: configurable 2000-entry compact discovery archive, historical
  region-best utility, diminishing visit-based novelty, region-aware retention,
  lazy eviction and cached ranking. Top 30 elites supply at most 16 structural
  imitation samples; a separate 30-entry conventional HOF supports the viewer.
- `policy.py`: reuse the structural encoder/head, add archive/budget context,
  freeze each on-policy cohort, and train structure rather than the rate head.
  Physical loss, exploration reward, entropy and imitation are separate quantities.
- `engine.py`: mixed RL/archive-edit/random seeds; bounded fitting and optional
  additional structural edits; deterministic proposal-order commits; explicit
  origin, parent, emitter, policy-version and parameter-operator metadata. Archive
  seeds inherit surviving rates; RL/random seeds start at rate 1. This difference
  is documented, not treated as a controlled source-comparison experiment.
- `harness.py`: retain existing Decider/Writer, memberwise validity, asynchronous
  scheduling, all communication arms, and request-zero HOF withholding. Additional
  exploration state is supplied as a workspace file only when outward conditioning
  is enabled and the request is not zero. Final candidate scoring uses the shared
  cache and budget; no new system prompts or provider implementation were created.
- `database.py`, `telemetry.py`: existing results schema with accurate RL/GA/RANDOM/
  LLM emitters, local metrics, cached-trajectory HOF images, progress/source plots,
  and Comet logging. Existing binary RL/LLM analyses need explicit adaptation.
- Durable checkpoints include policy, Adam, RNGs, archive, pending cohort and
  available Harness memory/provenance. Changes of code/config/evaluator fail
  resume validation. Interrupted external requests require reconciliation rather
  than automatic duplicate model calls.

## Verification on 2026-09-14

The broad check covers the two new test modules and existing parity evolution,
LLM integration and hybrid-runner configuration tests: **74 passed in 15.54 s**,
including the final compact-cache/provenance regression. Both real smoke manifests
match the final implementation source hashes.
Tests cover cache equivalence and bounds, failures and budget reservations,
concurrent duplicate calls, archive capacity/selection, no parameter gradients,
stale-policy rejection, memberwise Harness validity, communication isolation,
request-zero withholding, lineage, deterministic checkpoint continuation,
pending-cohort recovery, worker termination/replacement, database and Comet APIs.

Real CVODE checks used CPU only, two numerical workers, and no LLM calls:

| Run | Backend | Epochs | CVODE evaluations | Cache hits | Best loss | Fitting failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/comet-smoke-20260914` (Logic) | evolution | 2 | 24 | 0 | 0.4248515959 | 0 |
| `runs/rpa-powell-smoke-20260914` | Powell | 1 | 15 | 3 | 0.5353932981 | 0 |

Both completed successfully with finite policy gradients. Neither found a design
meeting its configured qualification criterion; these tiny checks are not evidence
for improved search performance. Final live archive rate ranges were
`[0.0061236075, 10.3737682853]` for Logic and `[0.0151474626, 1.2307502814]`
for RPA, within the configured `[0.001, 100]` bounds.

The user explicitly approved the Logic smoke upload to `genai-net-exploration`.
Comet completed its upload and exited cleanly:
https://www.comet.com/redsnic/genai-net-exploration/84bf0b726c1b4dcd8edf3bdf87d8d715

Uploads contain configuration, metrics and plots. Automatic code, Git metadata,
Git patches, host, CPU and GPU logging are disabled. No API key was placed in
source, commands or artifacts. RPA was local-only. PNG/PDF artifacts were produced;
the Logic progress image was visually inspected. Epoch plot indices are zero-based.
The Logic compatibility DB contains eight evaluated fitted designs and three HOF
snapshots, separately from its 24 physical evaluation records in the cache.

The initial development smoke `runs/smoke-20260914-a` failed on an incorrect
output-label attribute in the new adapter. It was fixed to use the actual task API;
the failed artifact was retained. Earlier development smokes are not final-source
verification runs and have not been relabeled or silently overwritten.

Removing repeated full trajectories from processing, event, design and cohort
records reduced the Logic smoke ledger from 62,091,264 to 18,391,040 bytes.
These are engineering storage observations, not paired scientific outcomes; the
canonical fitting-order change also changed the sampled rate-fitting trajectory.
Full diagnostics remain compressed in the evaluation cache and are restored on
replay. The live archive itself contains compact records.

## Final Audit and Remaining Limits

- Legacy checksum verification passes for all 214 frozen files. No old runs,
  manuscript results, defaults, prompts or source files were replaced.
- Real CPU evaluation and live Comet upload are verified. Harness wiring is covered
  by integration tests, not a paid end-to-end model call in this implementation.
- Large archive capacity is tested as a data structure, not yet benchmarked under
  a long scientific campaign. Full trajectory storage still grows with evaluations:
  the Logic smoke uses about 18 MB for 24 evaluations. Check available disk before
  scaling; this is not a constant-space cache. Policy/Adam checkpoints are about
  394 MB (Logic) and 382 MB (RPA) with the inherited architecture.
- Descriptor role counts are a pilot proxy, not mechanistic diversity. Historical
  successful-region counts and live archive counts are distinct. Descriptor choice,
  reward scale, fitting budgets and success criteria require scientific review.
- Cohort updates wait for their own numerical jobs, preserving on-policy gradients.
  Harness requests remain asynchronous; no promise of fully asynchronous RL is made.
- Processing exclusion is specific to initialization, optimizer, seed and budget,
  not permanent exclusion of a topology. Failed fits remain retryable. A different
  optimizer/budget requires an explicitly distinct configuration/run.
- This release has fixed source fractions, no multi-parent crossover, no adaptive
  scheduler, no target-region-conditioned policy, no SSA, and no CVODES sensitivity
  or neural ODE optimizer. Optional structural edits are bounded single-parent edits.
- The legacy `cumulative_rl_evaluations` field represents combined non-LLM fitting
  work in this method; use the detailed ledger for fair budget comparisons.
- Metrics can retain an uncommitted epoch after a crash; plot regeneration selects
  the latest record per epoch. Evaluation reservations remain conservatively charged.
  Arbitrary power-loss equivalence or external-model completion replay is not claimed.
- No long campaign was launched. Next action is to review pilot settings and run a
  bounded pilot before choosing a frozen comparison protocol.

## 2026-09-14: Behavior-preserving optimization pass

The user approved all five bookkeeping optimizations and the tensor/NN-path
optimizations, retaining the 10240-dimensional intermediate representation.
The pre-optimization implementation was preserved in
`baseline/exploration-pre-optimization-20260914.tar.gz`. Existing campaign outputs,
legacy sources, prompts and manuscript results were not edited.

### Implemented changes

1. **Checkpoint writes:** `engine.py` now writes a small `pending.pt` journal
   against a durable model checkpoint generation, rather than serializing unchanged
   policy/Adam tensors before every cohort. Full checkpoints remain initial,
   post-update and explicit terminal saves. Journal and checkpoint writes fsync
   their files and parent directories. Stale journals cannot override newer
   committed model states. A pre-Harness-dispatch intent also fails closed on
   interruption instead of silently repeating a possible external request.
2. **Archive selection:** `archive.py` freezes parent groups and visit-weighted
   probabilities once per cohort. `store.py` fetches counts in bounded SQL batches.
   Random draw order and probabilities match the prior implementation.
3. **Lock contention:** cache reads fetch bytes under the lock, then deserialize
   and decompress outside it. Coalesced callers can independently request full or
   summary payloads. The physical evaluation is still performed only once.
4. **Durable SQL batching:** proposal counts, archive ledger updates, fit-state
   records, reservations and completed evaluation records use logical transactions.
   Reads no longer execute an INSERT. Reservation commit still precedes dispatch,
   completed summary/diagnostics are atomic, and failed evaluations remain charged.
   No durability setting or timeout was relaxed.
5. **Lazy trajectories:** new `evaluation_summaries` and `diagnostics` tables keep
   scalar loss/validity/identity separate from compressed full task information.
   Fitting and replay use summaries; the trainer loads a selected fitted design's
   diagnostics when recording it, one design at a time. Existing monolithic cache
   entries remain readable. No precision reduction, key rounding or trajectory
   deletion was introduced. The long-run disk-capacity caveat still applies.
6. **Tensor rollout:** `policy.py` obtains the initial explicit observation once,
   maintains batched presence/parameter tensors, and updates selected slots without
   cloning environments or recompiling partial CRNs. Action IDs return to Python
   once per trajectory batch, not once per action step. Masks and nonfinite checks
   remain in place.
7. **Replay:** sampled observation prefixes are retained without gradients for
   on-policy replay, guarded by sequence and policy version. Gradients are freshly
   computed; imitation prefixes are likewise tensor-based. Resume rebuilds prefixes
   deterministically from saved action sequences rather than checkpointing them.
8. **Unused parameter head:** new `session.py` composes the existing session APIs
   with no real rate-head allocation. A meta-device head defines initialization
   shapes; at most 65536 CPU scratch elements consume the old RNG draws. Structural
   initial weights and seeded RNG state match the old builder. The runner uses this
   builder; externally supplied legacy Sessions remain supported without deleting
   their owned modules.

New tooling: `benchmark_optimizations.py` produces local JSON and PNG/PDF timing
artifacts. The two exploration test modules now cover all of the above paths.

### Verification

**85 tests passed in 16.65 s** across the exploration, parity-evolution, LLM
integration and hybrid-runner configuration suites. Added checks cover:

- Exact CPU equality of Logic/RPA observations, actions, log probabilities,
  entropies, gradients and seeded sampling against the original environment path.
- Structural-only session initialization weights and RNG equality, including
  scratch-buffer chunk boundaries; no rate head is allocated.
- Lazy cache reads, decoding outside the lock, mixed full/summary coalescing,
  legacy payload reads, transaction rollback and simulated failed completion.
- Identical archive-parent random draws and durable pending-cohort recovery,
  including bitwise policy continuation, stale journal rejection/ignoring, and
  interrupted Harness-dispatch reconciliation.
- Existing memberwise Harness validation, communication isolation, request-zero
  HOF withholding, CVODE timeout handling, provenance, database and Comet interfaces.

Fresh real CVODE smoke artifacts:

| Task | Output directory under `runs/` | Physical evaluations | Cache hits | Best loss |
| --- | --- | ---: | ---: | ---: |
| Logic / evolution | `optimized-logic-smoke-20260914` | 24 | 0 | 0.42485159588611987 |
| RPA / Powell | `optimized-rpa-powell-smoke-20260914` | 15 | 3 | 0.5353932980534578 |

Both finished without fitting failures and reproduced their pre-optimization
losses. Both terminal policies and Adam states are bitwise identical to their
pre-optimization counterparts, and both discovery archives are identical. Both
smoke manifests match the final source hashes. All 214 frozen legacy source files
still pass checksum verification. The Logic pending journal
is 19,753 bytes, versus a 393,829,141-byte full terminal checkpoint. This reduces
write volume; the full checkpoint has not become smaller. No new LLM calls or
external uploads were made in this pass; Comet integration remains tested.

### Measured CPU timings

Artifact: `runs/optimization-benchmark-20260914/benchmark.json` and
`policy_timing.png/pdf` (visually inspected). One CPU thread, 16 Logic proposals,
16 imitation examples, five reaction steps, 10240-dimensional intermediate layer,
three repetitions after a warm-up per implementation:

| Operation | Environment path | Tensor path |
| --- | ---: | ---: |
| Median sampling time | 0.2102 s | 0.0848 s |
| Median update time | 1.3249 s | 0.9293 s |
| Median sampling + update | 1.5321 s | 1.0143 s |

The combined NN path takes about 34% less time (1.51x throughput); sampling alone
is about 2.48x faster. A separate 16000-value cached trajectory payload took about
3.57 ms per full read versus 0.012 ms per summary read. These figures exclude
CVODE, checkpoint writes, Comet and external model latency; they do not establish
an end-to-end campaign speedup. No GPU timing claim is made: `nvidia-smi` could not
communicate with the NVIDIA driver in this session. The GPU driver was not changed.

The 1024-dimensional alternative is deferred, as requested. Learning objectives,
success cutoffs, rate bounds, solver tolerances, archive sizes and fitting budgets
were not changed. A source fingerprint change prevents silently resuming older
scientific runs under the optimized implementation.

## 2026-09-14: Storage cleanup; NN-size benchmark paused

The requested 1024 versus 10240 NN timing comparison was paused at the user's
request before any benchmark workers started. A separate benchmark script was
drafted, but it has not yet been tested or executed; no size-comparison results
are available. The production architecture remains unchanged.

The sandbox initially failed with ENOSPC. `/local0` had about 740 MiB available;
inodes were not exhausted. With the user's cache-cleanup authorization:

- Ran the environment's pip cache purge on
  `/local0/home/rossin/.cache/pip`, removing 813 cache files.
- Pip 22.0.2 did not purge the newer `http-v2` download cache. Removed only
  `/local0/home/rossin/.cache/pip/http-v2` and the old `http` download-cache directory.
- The pip cache decreased from about 23 GiB to 208 KiB; `/local0` now reports
  24 GiB available. Installed packages, virtual environments, model files, Harness
  files, experiment outputs and checkpoints were not removed.
- `/local1` has about 1.7 TiB available. `/local1/rossin` does not yet exist;
  no directories or data were moved there during this cleanup.

All 214 frozen legacy files still passed verification during the cleanup.
The benchmark remains paused pending continuation.

## 2026-09-14: NN-size timing completed and quality pilot launched

The user resumed work after creating `/local1/rossin`. The isolated CPU timing
benchmark completed all eight cases at
`/local1/rossin/search-exploration/nn-size-benchmark-20260914`. Each case used
two warm-ups and five measured updates in a fresh process, fixed FP32, hidden
width 1024, depth 5, and 16 matched imitation examples. Results are medians:

| Task | RL samples | 1024 | 10240 | Large/small time ratio |
| --- | ---: | ---: | ---: | ---: |
| Logic | 16 | 0.275 s | 0.959 s | 3.48 |
| Logic | 128 | 0.745 s | 2.020 s | 2.71 |
| RPA | 16 | 0.257 s | 0.930 s | 3.63 |
| RPA | 128 | 0.695 s | 1.975 s | 2.84 |

Logic has 13,894,053 versus 32,814,501 trainable parameters; RPA has 12,879,963
versus 31,800,411. Parameter/gradient/Adam storage is 212.0 versus 500.7 MiB for
Logic and 196.5 versus 485.2 MiB for RPA. These exclude activations/framework
overhead; peak process RSS is separately recorded. No search-quality conclusion
follows from synthetic-reward timing. The executed script is preserved beside
the artifacts. A later legend-only rendering adjustment is recorded separately
in `rendering.json`; raw timing measurements were not rerun or changed.

The user clarified that actual search quality also needs testing. Added:

- `run.py`: explicit positive `--embedding-size`, frozen/logged effective policy
  width, depth and intermediate dimension, plus elapsed time in final summaries.
  No default size change. This creates a new source identity, not a silent change
  to old runs.
- `nn_size_quality.py`: frozen paired pilot configurations, separate storage,
  bounded concurrency, source hashes/archive, Comet logging, controller lock,
  durable progress records, and fail-visible queue holding without retries.
- `analyze_nn_size_quality.py`: complete-pair endpoint loss and topology counts,
  paired W/T/L and descriptive ratios, observed convergence versus actual
  evaluations, fit-failure totals and explicit unscorable pairs. No significance
  or equivalence claim is made for the five-seed pilot.
- `test_search_exploration_nn_sizes.py` and
  `test_search_exploration_quality_pilot.py`: matching/size validation, aggregation,
  missing-pair behavior, shared budgets, no Harness calls and explicit Comet flags.

Verification: **96 tests passed in 16.79 s** across these modules and the existing
exploration, parity evolution, LLM integration and hybrid configuration checks.

Actual search pilot:

- Root: `/local1/rossin/search-exploration/nn-size-quality-20260914`.
- tmux: `nn-size-quality-20260914`.
- Tasks: Logic and RPA; seeds 0..4; dimensions 1024 and 10240; 20 runs total.
- Twenty epochs, batch 32, up to 16 evolution rate-fit calls per proposal, 10240
  physical-evaluation cap, four CVODE workers per run, and 12 concurrent runs.
- Bounds `[0.001,100]`, fit/evaluation deadlines 60/30 s, archive capacity 2000,
  top-30 imitation, at most 16 demonstrations; no LLM or RAG calls.
- Same task/seed settings file for both sizes. Logic retains all-row qualification
  and its 0.04271267898491677 cutoff; RPA uses 0.016491168958935352.
- Comet: `genai-net-exploration`, with the existing privacy restrictions. The first
  12 experiments registered successfully; eight jobs remain queued at launch.
- Primary interpretation concerns quality, not just speed. Equal caps/schedules
  do not imply identical consumed budgets, so actual counts and evaluation-axis
  curves are reported. No quality outcome is claimed at launch.

## 2026-09-14: RPA 100-epoch exploration run without LLMs

The size pilot completed all 20 runs with no run or recorded fitting failures.
The user selected the smaller network for a more substantial RPA experiment.
No default architecture or scientific implementation was changed.

- Added `configs/rpa-100epoch-no-llm.json` and `RPA_NO_LLM_RUN.md`.
- One seed (0), 100 epochs, batch 1023, NN intermediate dimension 1024;
  evolutionary rate fitting with 16 calls per proposal and a 1,636,800 physical
  CVODE evaluation cap. Same optimizer, bounds and qualification as the pilot.
- Source probabilities remain RL .5 / archive mutation .25 / random .25;
  this is not a legacy RL-only control. No Harness, LLM or RAG calls.
- 48 persistent CPU workers; GPU driver unavailable, left unchanged.
- Configuration and task snapshots plus a source archive saved under
  `/local1/rossin/search-exploration/rpa-no-llm-100epoch-20260914`.
  Separate `run/` output, `run.log` and tmux session of the same campaign name.
- Comet live experiment:
  https://www.comet.com/redsnic/genai-net-exploration/3971107d201b4cd18284fc550665237d
- Preflight validated the configuration, budget product, RPA evaluator and
  forbidden reaction IDs [0, 1, 28]. All 214 frozen legacy files verified.
  The four focused exploration test modules passed: 47 tests in 12.90 s.
- Existing runs and unrelated worktree edits were preserved. No automatic
  retries, manuscript edits or quality claims were made at launch.

## 2026-09-14: Independent search and optimization streams

User authorized implementing and evaluating the asynchronous design. Added the
separate `RL4CRN/search_streams/` package, `run_streams.py`, the detached completion
controller, saved-results analysis, artifact audit, a matched worker benchmark,
validation configuration and tests. Full protocol and caveats are in
`STREAMS_IMPLEMENTATION.md`.

- Direct-search cohorts update independently of fitting jobs. Delayed fitted
  candidates influence the archive, mutation parents and elite imitation, never
  ordinary on-policy credit for a changed policy.
- Whole rate-fitting jobs execute in workers. Compact scalar reservation/result
  messages replace per-evaluation full trajectory transfer. Every scalar outcome
  remains cached; rich diagnostics are retained for archive admissions.
- Bounded 512-job full-run queue; at most 256 admissions per epoch. Quality,
  descriptor-coverage and random selection are combined. Incomplete starting
  trajectories remain eligible for coverage/random fitting, not silently dropped.
- Exact cache keys, duplicate coalescing, failure accounting and FULL SQLite
  durability retained. Optimization cannot consume the reserved search budget.
- Viewer provenance distinguishes direct search from rate-refined descendants.
  Existing scalar evaluator consumers have a single-charge compatibility adapter.
- New CLI has no LLM launch flags. Comet retains privacy settings and logs metrics
  and figures. Automatic asynchronous resume is refused pending reconciliation.

Verification: **53 tests passed in 19.25 s**. The real CVODE fitter reproduced
reference best records; delayed-optimizer tests verified continued RL updates;
budget, cache, provenance and incomplete-start eligibility tests passed. All
214 frozen scientific files remain unchanged. Current source hashes also match
both the independently running synchronous and asynchronous full-run manifests.

Completed matched benchmark: 32 frozen proposals x 16 calls, eight workers;
512 evaluations each, exact best-record/call-count equality, zero fitting
failures. Full-transfer path 19.154 s versus compact worker-local fitting 10.794 s
(1.77x throughput). SQLite sizes 131,665,920 versus 487,424 bytes, respectively.
The compact benchmark retains scalar records only; actual training also retains
archive diagnostics. This is one timing sample, old path first, not a statistical
speed comparison. Artifact: `/local1/rossin/search-exploration/stream-worker-benchmark-20260914`.

Two end-to-end 20-epoch, batch-32 validations completed, including queue draining:

- Initial validation: 2335 evaluations, 126.81 s, best loss 0.15266600, zero
  qualifying topologies, zero fitting failures.
- Incomplete-start safeguard validation: 2271 evaluations, 132.21 s, best loss
  0.15459799, zero qualifying topologies. All 109 dispatched jobs merged; 106
  improved starting loss. Read-only audit reconciled all charges and source
  snapshot hashes, found no out-of-bound recorded designs or runtime failures,
  and explicitly counted 150 incomplete numerical trajectories.
- Existing same-seed synchronous pilot: 10200 evaluations, 747.45 s, best loss
  0.15750348, zero qualifying topologies. Different budgets/schedules and machine
  load preclude interpreting this as statistical superiority or pure speedup.
- Final validation comparison: `/local1/rossin/search-exploration/stream-validation-final-comparison-20260914`.

Full evaluation launched in tmux `rpa-streams-evaluation-20260914`:
`/local1/rossin/search-exploration/rpa-streams-100epoch-20260914`.
Same seed 0, 100 epochs, batch 1023, intermediate NN size 1024, canonical RPA,
bounds and total evaluation cap as the reference; 32 search + 16 optimizer workers.
The controller waits for all asynchronous work to finish, generates a comparison,
then refreshes it when the original synchronous run finishes. No reference run
is stopped or restarted. Comet experiment:
https://www.comet.com/redsnic/genai-net-exploration/704919d5c66f48689a25cdec7ca4e88f

Storage incident: `/local0` filled during a new-file write. The partial new runner
was zero bytes and was subsequently written completely. Completed smoke artifacts
(2.6 GiB) were moved intact to `/local1/rossin/search-exploration/completed-smokes-20260914`;
their former repo path is a symlink. No results or model files were deleted.
The already active RPA run wrote to `/local1` throughout and was not interrupted.

### Full Asynchronous Evaluation Completed

All 100 epochs and all 8666 dispatched fitting jobs completed. Elapsed time was
1397.58 s (23.29 min), including initialization. Final loss 0.002742564523021603;
57 qualifying topologies in the 2000-entry discovery archive. Physical evaluations:
206870 total = 77235 search + 129635 optimization. Cache hits 34068; coalesced
in-flight lookups 18. There were 8475 jobs improving their starting loss.

Final read-only audit reconciled every charge and dispatch/merge, verified source
snapshot hashes and evaluator identity, and found no runtime failure events,
malformed recorded reaction counts or out-of-bound recorded rates. It explicitly
counted 15117 incomplete numerical simulations and 15071 retained diagnostics.
No incompleteness was relabeled as a successful simulation.

The generated comparison PNG was visually inspected. At that snapshot the
synchronous reference was still at 22 epochs, 352241 evaluations, loss 0.00571082
and four qualifying topologies. Its last recorded boundary below the shared
206870-evaluation limit was 195603 evaluations, loss 0.05095534 and zero qualifying
topologies. The boundaries are not exactly matched and the algorithms differ;
this remains one-seed exploratory evidence. The original reference continues
unchanged and the controller will regenerate the comparison when it finishes.

Report: `/local1/rossin/search-exploration/rpa-streams-100epoch-20260914/analysis/REPORT.md`.

### Final Review: Prototype Archive Eviction Defect

A targeted min-heap regression found that the inherited exploration archive used
negative utility in its eviction priority. Among non-representatives it could
evict a qualifying design before an unqualified one. A two-slot example retained
losses 0.001 and 1.0 while discarding 0.002. This is in the new exploration
prototype, not a change to the canonical evaluator or pre-existing paper cohorts.
The v1 run's ledger has 137 discovered qualifying topologies but only 57 retained
in its final archive. Its outputs were preserved, not rewritten or hidden.

Added `StreamDiscoveryArchive` under the separate new namespace, correcting the
utility sign while retaining representative protection. No synchronous or legacy
source was edited. Version 2 is explicitly recorded in the new manifest, Comet
tags, database metadata and state. The analysis discloses the extra retention
policy difference instead of attributing its effects solely to asynchrony.

54 tests passed in 201.59 s under concurrent workload, including archive ranking
and state reconstruction. Launched a fresh complete v2 evaluation with identical
settings under `/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914`;
tmux `rpa-streams-v2-evaluation-20260914`. Comet experiment:
https://www.comet.com/redsnic/genai-net-exploration/3946a0d47774489dbb434b9c122e9818
The original synchronous reference and completed v1 outputs are preserved.

`replay_archive_retention.py` replayed the complete v1 discovery sequence with no
new evaluations. The original terminal membership matched exactly, validating
the reconstruction. Corrected retention kept 137 qualifying topologies versus
57 under the old rule on those identical records. This is a retention-only
counterfactual, not a claim about the corrected run's learned policy. Artifacts:
`/local1/rossin/search-exploration/stream-archive-retention-replay-20260914`.

### Corrected V2 Evaluation Completed

Completed 100 epochs and drained all 8951 fitting jobs in 1685.01 s (28.08 min).
Best loss: 0.0025487682658084914. Retained 176 qualifying topologies in the
2000-entry archive, spanning 55 qualifying descriptor regions. There were 8726
fits improving their starting loss. Physical evaluations: 214793 = 80887 search
+ 133906 optimization; 30707 cache hits and 16 coalesced in-flight lookups.

Final audit passed frozen-source/evaluator checks, budget reconciliation, complete
dispatch/merge reconciliation, recorded rate bounds and structural validation.
Zero runtime failure events and zero pending jobs. Incomplete simulations remain
visible: 15203 scalar outcomes. Rich diagnostics were retained for 15235 evaluations.
No model calls were made and no manuscript files were changed.

Visually inspected the generated comparison. The still-running synchronous
reference had 26 epochs, 413141 evaluations, loss 0.005443852218968288 and 11
qualifying topologies at the report snapshot. Its latest checkpoint below the
shared 214793-evaluation limit was 211865 evaluations, loss 0.0509553354751609 and
zero qualifying topologies. These are not exactly matched budgets or a pure
asynchrony ablation; the retained reference also uses the original archive rule.
All old results and implementations remain preserved. The controller will update
the comparison after the original reference finishes.

Report: `/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/analysis/REPORT.md`.

Post-run compatibility fix: requesting diagnostics for a scalar-only cache entry
now returns `task_info: {}` rather than omitting the key, supporting inherited
diagnostic sampling. Added assertions for both diagnostic and scalar-only access.
No recomputation is introduced and scalar worker/training behavior is unchanged.
The run's frozen source archive records the exact executed version before this
compatibility-only change.

Final verification after that compatibility fix: 54 focused tests passed in
19.68 s, all 214 frozen legacy scientific files passed checksum verification,
and scoped `git diff --check` reported no whitespace errors.

### 2026-09-15: No-RL Control And Recorded Interaction

Added an isolated no-RL control of corrected asynchronous v2, replacing its 50%
RL share with random proposals while preserving the 25% archive-mutation share.
StreamTrainer now skips both policy and imitation updates when `rl_fraction=0`;
positive-RL execution is unchanged. A regression fails if either the learner
update or imitation sampler is called in no-RL mode. The new manifest records
the proposal arm and whether policy training is enabled. Comet identifies the
control explicitly. Legacy exploration and evaluator sources remain unchanged.

Added `run_no_rl_control.py`, copying frozen reference settings to a fresh folder,
logging exact commands/differences and progress, and generating an audited
comparison on completion without retries. `analyze_streams.py` now distinguishes
no-RL arms and does not incorrectly attach synchronous-retention caveats to a
comparison of two corrected asynchronous runs.

Added `analyze_source_interaction.py`: event-order first qualification attribution,
overlapping per-source topology counts, and explicit mutation origin/parent-loss
analysis. It uses event provenance rather than mutable latest-design provenance
and counts repeated topologies only once for first discoveries. Regression covers
cross-source rediscovery, fitting without topology change and mutation parentage.
Reference result: 139 direct RL first qualifications, 30 after fitting RL
proposals, 7 direct mutation, zero random; all 8 qualifying mutation events have
RL origin. No evaluator/model calls were needed for this analysis.

Verification: 55 tests passed in the five exploration suites before adding the
provenance regression; all 9 stream tests including that regression subsequently
passed. All 214 frozen legacy checksums verified. Scoped whitespace check passed.

Launched the 100-epoch, batch-1023, seed-0 control with 32 search / 16 optimizer
workers in tmux `rpa-no-rl-control-20260915`, isolated under
`/local1/rossin/search-exploration/rpa-streams-no-rl-100epoch-20260915`.
Comet: https://www.comet.com/redsnic/genai-net-exploration/92089b8f53cc453284c62a284ec31a05
Completion must be checked from run artifacts; this entry records launch only.
Protocol, interpretation and exact command: `RPA_NO_RL_ABLATION.md`.

Launch verification: frozen task, archive rule, queue and worker settings match
the completed v2 reference; the sole ExplorationConfig difference is
`rl_fraction: 0.5 -> 0.0`. The first epoch completed with zero RL samples,
zero imitation samples, `Policy/enabled=0`, and policy version zero. All 56
focused tests passed together in 20.48 s after the provenance regression.

### 2026-09-15: Reciprocal Control And Standalone GA

User approved the remaining comparison arms. Added no-archive-mutation naming
and configuration support; the control replaces only the 25% structural mutation
share with random proposals, keeping 50% RL and the independent rate fitter.
No default scientific settings or older implementations were changed.

Added `bounded_ga.py` and `run_standalone_ga.py`, importing the original mutation
GA unchanged, with an opt-in post-proposal rate clip. Uses original 45/45/10
operators, top-30 parents, [0.1,50] initialization prior, batch 1023 and no LLM
replay. Target is the hybrid's 214793 actual physical evaluations, not 100 GA
batches. Cache hits, proposal counts, clipping, incomplete simulations and all
designs are logged; selected diagnostics and HOF snapshots use viewer-compatible
SQLite. Comet metrics and plots are enabled. A finite proposal safety cap fails
explicitly rather than silently declaring an under-budget run completed.

Extended `analyze_streams.py` with named arms, stable colors, four-arm support
and `--discovered` full-ledger unique qualifying topology curves. Reporting the
whole discovery history avoids comparing a 30-slot reporting cap with 2000 slots.
It does not remove algorithmic effects of those parent archives. Extended the
artifact audit for exact-budget standalone termination and admissible IDs.

61 tests passed in 20.20 s; all 214 frozen legacy source hashes passed. Real
CVODE smoke completed 64 evaluations with zero runtime failures and eight
explicit incomplete simulations. Its final source/budget/structure/bounds audit
passed. Read-only SQLite WAL access required host permissions; no database
contents or security controls were changed. Plotting smoke was generated and
visually inspected; it is labeled software verification, not comparative evidence.

Launched `run_remaining_controls.py` in tmux `rpa-remaining-controls-20260915`:
two concurrent jobs, 48+16 evaluator workers, normal priority, CPU only. Existing
hybrid and no-RL results are reused. Campaign root:
`/local1/rossin/search-exploration/rpa-remaining-controls-20260915`.
Exact protocol, commands, Comet links and accounting caveats are in
`RPA_FOUR_ARM_COMPARISON.md`. No manuscript edits or new model calls.

Post-launch verification: no-mutation differs from the reference config only in
`mutation_fraction: 0.25 -> 0.0`; task configurations match exactly across new
arms. The standalone manifest confirms budget 214793, batch 1023, zero replay
events, bounds [0.001,100] and 30 parent entries. Both jobs advanced beyond
initialization. A full-ledger regression verifies that eviction does not erase
discovery counts and rediscovery does not double count a topology. Final focused
verification: 62 tests passed in 24.07 s. An initial four-arm report is saved in
the new campaign's `analysis/`; the controller regenerates it after completion.

### 2026-09-15: Test-First Persistent Julia Rate Optimizer

Added the opt-in `RL4CRN/parameter_optimization/` module, without changing any
search stream, canonical evaluator, campaign, or sibling SynthEvo source.
AdaGrad is the default; Adam is optional. The worker includes SynthEvo's existing
`CRNOptim.jl` update helpers but implements a general numerical forward-sensitivity
mass-action kernel instead of the IFF-specific symbolic optimization loop.

The interface separates CRN structure from experiment data. A starter/library
adapter precomputes reaction/species/input metadata once and selects reactions in
explicit parameter order. It preserves input modulation and repeated-reactant
power conventions, excludes starter-overlapping and empty library reactions, and
does not compile a symbolic network for every selection. General CRNSpec is not
itself a benchmark validator. Experiments currently cover independent constant-
input scenarios, linear outputs, and weighted absolute/squared trajectory losses.

One persistent Julia process executes complete multi-start jobs. `submit` returns
a Future and snapshots queued inputs. There is no automatic search integration
or process pool. Prepared models use a bounded LRU cache; exact continuation
restores accumulators/moments, step, current gradient, and best evaluated state.
Checkpoints include backend/source/Manifest fingerprints and corruption checks.
Log-rate updates project onto [0.001,100]; fixed rates are unchanged. Failed
starts and known solver failures are separate outcomes. Unknown exceptions
propagate; hard deadlines stop/reap the worker rather than fabricate losses.

Tests were written before implementation. Final verification: 14 optimizer tests
including actual Julia/CVODE solves, plus 15 existing stream/standalone-GA tests,
all passed (29 total, 28.33 s). All 214 frozen legacy source hashes still match.
Tests cover analytic and finite-difference gradients, multi-output/relative loss,
zero states, repeated reactants, ordering, canonical RPA trajectories, clipping,
independent restart failure, exact continuation across worker processes, cache
eviction, and killable deadlines. No new Python/Julia packages were installed;
the existing Julia user compilation cache was used. Runtime was Julia 1.12.1,
Sundials 5.2.0, SciMLBase 2.155.0 (not the Julia version of older SynthEvo reports).

Two useful discrepancies were caught rather than hidden:

1. A reference-test IOCRN reused its cached trajectory between scenarios. The
   fixture now explicitly resets it; production IOCRN behavior is unchanged.
2. The first fitting probe omitted the canonical all-species 10,000 ceiling.
   Two apparent low-output-loss optima had peak species 10466.6 and 10675.0;
   canonical CVODE correctly rejected them. The full diagnostic is preserved in
   `reports/julia-rate-prototype-20260915-diagnostic/`. These are NOT valid gains.
   Added an explicit experiment ceiling, continuous terminal event, and both a
   toy mixed-restart regression and the actual offending RPA vector. The general
   interface requires callers to supply domain-specific ceilings explicitly.

The guarded verification is in
`reports/julia-rate-prototype-20260915-guarded/{REPORT.md,results.json,rate_fitting.png}`.
With precompiled dependencies, worker startup was 1.461 s, first preparation
1.284 s, first joint evaluation 2.086 s, and warm loss-plus-gradient evaluation
0.027 s for all nine 1000-time-point RPA scenarios on one Julia/BLAS thread.
Maximum initial output discrepancy from canonical Python CVODE was 5.24e-7.
AdaGrad improved the perturbed start from 0.820645 to 0.00391752 before stopping
at the species ceiling; Adam retained 0.110233. All four trials stopped early at
the guard, and all four retained best vectors passed independent canonical
reevaluation. This is one-topology engineering evidence, not optimizer superiority
or a completed comparative campaign. The original good start retained its best.

The guarded run's metrics, status, figure, and report were logged to Comet project
`genai-net-exploration`, experiment `7504ec016f0740e5812db2ea9ab8af75`.
The initial aborted experiment `a59da44cf9c74e479d98fee39ce8c1a2` is documented
separately and is not merged into valid results. Reports can be regenerated with
`benchmark_julia_rates --render-only` without any model calls or optimization.

Remaining gates before integration: feasibility-aware rate-step handling or a
deliberately conservative step-size study, multiple topologies and starts,
comparison with existing fitters under measured compute budgets, then asynchronous
search integration. Julia augmented-ODE adaptive steps and saved-state positivity
checks are not identical to Python's state-only solver/inequality constraints;
canonical reevaluation remains mandatory. No manuscript results were changed.

### 2026-09-15: Feasibility Backtracking And Matched-Start Test

Added opt-in `step_policy="feasible"` in a separate `feasible.jl`; the original
`optimize_one` stop-on-failure loop remains unchanged and is still the default.
Known infeasible trials halve the same log-coordinate update, up to a finite
retry cap. Tentative AdaGrad accumulators/Adam moments are committed once on
acceptance, not accumulated repeatedly on rejection. This tests feasibility,
not sufficient loss decrease. Unexpected exceptions still propagate.

Added per-start/per-call evaluation caps, explicit rejected-trial records,
accepted-step histories with actual evaluation/solve/time accounting, and
backtracking-exhausted/budget-exhausted/stalled outcomes that preserve the last
valid state. The added source is checkpoint-fingerprinted. No campaign,
canonical evaluator, manuscript, or sibling SynthEvo source was changed.

Four new tests (one parameterized for AdaGrad/Adam) verify retries without
reaccumulation, rollback, per-element initial failure, budget limits, and exact
continuation. Final focused suite: 33 passed in 28.72 s (18 optimizer + 15 prior
stream/GA tests). All 214 frozen scientific source hashes still match.

Ran `comparisons.search_exploration.test_feasible_rates` locally with eight
predefined rate starts on the same RPA topology, 32 requested updates each:
stop/lr=0.1, stop/lr=0.01, and feasible/lr=0.1 (eight halvings, at most 65 joint
evaluation attempts). Arm order rotates across starts and warmup is separate.
The same update cap does not imply equal consumed compute; all failed partial
scenario solves and wall costs are reported. Five of eight initial starts are
infeasible in every arm and were not resampled or dropped.

Artifacts: `reports/julia-feasible-rates-20260915/` contains raw results and
checkpoints, rejected trials, endpoint CSV, report, and a three-panel plot of
all initially feasible starts. Curves carry the retained best through final
rejected trials so the plotted endpoint includes their cost. All nine retained
best rate vectors passed independent canonical completeness/loss verification.

Main observations:
- Difficult perturbed start: stop/lr=0.1 retains 0.00391752 in 13 attempts,
  115 scenario solves, 0.381 s. Backtracking retains 0.00153414 in 50 attempts,
  352 scenario solves, 1.085 s, but exhausts retries after 21 accepted updates.
- The already-good start does not improve, and backtracking exhausts retries.
- On the third feasible start, stop/lr=0.01 is best (0.000790929 vs 0.00141238
  for both lr=0.1 variants); all three complete 32 updates there.
- Overall completions: 1/8, 2/8, 1/8. Valid retained bests: 3/8 in every arm.
  Total joint attempts: 53, 94, 147; scenario solves: 433, 810, 1012.

This supports a narrow utility claim for bounded backtracking, not universal
superiority or readiness for search integration. It exposes a need for feasible
initialization and better step selection near the concentration boundary. It
does not establish multi-topology performance or equal-compute superiority.

Comet upload was blocked by automatic approval: previous authorization was judged
to cover the earlier smoke only, not this broader comparison. Explicit approval
for uploading this comparison's metrics/plot/report was requested. The test then
ran without Comet; no external upload was attempted via another route. All
artifacts remain local and can be uploaded later without rerunning optimization.

### 2026-09-15: Read-Only Fitter Efficiency Audit

Added local profiling scripts `profile_julia_rates.py/.jl` and saved results in
`reports/julia-efficiency-20260915/`. No optimizer/evaluator/worker code changed,
no campaign started or stopped, and no Comet upload was attempted. The separate
Julia profiler loads the existing worker with EOF stdin and measures unchanged
evaluation, a solver-only diagnostic with identical augmented solves/events,
an isolated RHS loop, and a sampling profile. Python measures warm IPC, checkpoint
roundtrips, prepare/cache cost, canonical validation, and worker RSS. Timings are
medians over 20 repetitions except RPC (100) and Python canonical calls (10).

Observed on the reference RPA rates, all nine scenarios, one Julia/BLAS thread:
- No-op IPC: 0.0296 ms; cached checkpoint: 0.2193 ms.
- Cached prepare: 2.124 ms; fresh structure preparation after warmup: 7.160 ms.
- Warm Python-to-Julia loss/gradient: 21.583 ms; including trajectories: 25.791 ms.
- In-process full evaluation: 20.957 ms, 20.637 MiB allocated per call.
- Solver-only diagnostic: 11.766 ms, 5.042 MiB allocated per call.
- 10,000 RHS calls at one state: 1.200 ms, zero allocated bytes.
- Canonical Python helper: 181.397 ms, including materialization/validation;
  this is not an isolated solver comparison and computes no gradients.
- Startup + first prepare + first evaluation: about 4.64 s with installed caches.
- Worker RSS after warmup/several models: 779.3 MiB; pool scaling untested.

Sampling also shows substantial time in the saved-state/loss/gradient loop.
A typed reduction helper is a justified next performance experiment; about 9 ms
and 15.6 MiB separate full evaluation from the solver-only measurement, though
independent timing/GC means this is not an exact additive decomposition.
The abstract ODEProblem container is a likely inference issue, not proven by
compiler diagnostics in this audit. IPC and the reaction RHS are not priorities.

Reanalyzed existing trial histories without recomputation: the already-good start
spent 46/59 attempts and roughly 0.762/1.146 s on rejected-trial intervals without
improving. The difficult start spent 28/50 and roughly 0.448/1.085 s there. These
intervals include bookkeeping. Retrospective last-improvement positions are not
an early-stopping prescription; another start improved at its last evaluation.
All 214 frozen legacy source hashes and whitespace checks still pass.

### 2026-09-15: Explicit No-Pruning Contract

Confirmed that fitting imports only SynthEvo's numerical AdaGrad/Adam update
helpers. Neither rate sparsity regularization nor pruning is imported or enabled.
The L1 option is absolute trajectory-tracking error, not a rate penalty. User
prefers to reserve simplification for a separate future postprocessing experiment.

Documented this distinction and the prospective postprocessing protocol: preserve
the original design, protect template reactions, explicitly remove candidate
reactions, optionally refit surviving bounded rates, and canonically reevaluate
against a predefined performance tolerance with separate provenance and cost.
No postprocessing implementation or campaign changes were made.

Added a real Julia regression for both step policies using an uncoupled, unobserved
reaction. Its small/large rate contributes neither loss nor gradient and remains
present and unchanged after optimization, including at the lower rate bound.
All 20 optimizer tests passed in 22.89 s; all 214 frozen source hashes still match.

### 2026-09-15: Opt-in Asynchronous Julia Fitting and Separate Pruning

Implemented the approved pilot without changing frozen legacy sources, previous
outputs, the default evolution/Powell backend, the canonical evaluator, or the
main manuscript. `run_streams --julia-project ...` selects the new backend;
`--post-prune` enables a separate terminal simplification stage. The old config's
optimizer field remains for compatibility; manifests, Comet parameters and
candidate provenance record the effective `julia_adagrad` operator explicitly.

New modules in `RL4CRN/search_streams/`:
- `julia_adapter.py`: executable RPA/tracking and Logic task-to-Experiment
  adapter, including inputs, IC ordering, target, time weights, tolerances and
  the concentration ceiling. Non-L1, SSA and unsupported objectives fail closed.
- `julia_fitting.py`: bounded four-start screening, best-two continuation,
  per-topology accounting, complete-screen caching and resumable checkpoints.
- `julia_stream.py`: eight persistent Julia workers, bounded admission queue,
  50/25/25 admission mix, at most 32 distinct topologies per batch, 24 joint
  attempts plus optional 24-attempt extension for at most eight eligible
  asynchronous completions per submission epoch. Canonical verifications also
  count against the global budget; direct search has protected credit.
- `pruning.py`: explicit added-reaction deletion, at most five qualified parents,
  at most one accepted deletion per parent, 12 total evaluations per trial and
  60 per parent including parent verification. Positive bounded rates remain;
  no L1 rate penalty, rate zeroing, or implicit threshold deletion. Originals
  stay unchanged; derived records, failures, checkpoints and costs are separate.

Extended the Julia API with an optional runtime `evaluation_limit`, outside the
checkpoint configuration. Existing `max_evaluations` semantics and default stop
policy remain unchanged. Resuming with less remaining credit does not reset
optimizer memory or recompute the saved current gradient. Screening uses exact
rate/model checkpoint caching; canonical evaluations retain the existing shared
cache. Repeated internal trial evaluations remain charged. Failed initial rates
never permanently blacklist a topology; future visits get new restart seeds.

The trainer never waits for pending fits during an RL update. Only terminal drain
waits; delayed fits affect future archive sampling/imitation, not old on-policy
rewards. Async campaign restart remains unsupported without explicit pending-job
reconciliation. Hard worker errors/timeouts fail visibly; unknown evaluations are
conservatively charged, not hidden as successful or free attempts.

Retained local metrics, archive plots, SQLite viewer compatibility and optional
Comet logging. Added separate joint/canonical cost reporting, source snapshots,
Julia/backend fingerprints, complete per-job JSON, and a local report renderer.
No external uploads or model calls were made; the earlier Comet approval
restriction remains in force. No existing campaigns were stopped or restarted.

Verification and completed artifacts:
- `reports/julia-stream-smoke-20260915/`: 3 x 128 proposals, eight Julia workers,
  70.3 s including startup/postprocessing. Five fitted topologies, four improved;
  archive best 0.183939924 -> 0.162559515. Initial admission cost estimate allowed
  35.1% measured fitting worker-time vs a 20% target. Preserved as calibration
  evidence; subsequent code admits just one cold calibration job before opening
  concurrency and initializes its estimate from that measured job cost.
- `reports/julia-stream-pilot-20260915/`: 3 x 1023 proposals, eight search and eight
  persistent Julia workers, 172.6 s including startup/drain/postprocessing.
  All 26 fitting jobs merged; 24 improved their inherited candidate. First-epoch
  archive best 0.201820847 -> terminal 0.152702627. Two initially incomplete
  inherited candidates became canonically complete (losses 0.421323789 and
  0.292518519). 4,124 evaluation attempts = 3,092 canonical + 1,032 joint;
  9,196 Julia scenario solves. Ten initial evaluation failures were retained
  within otherwise completed jobs. Measured fitting worker-time share 14.7%;
  terminal pending jobs and reserved evaluations both zero.
- Neither search pilot reached the quality threshold 0.05, so no search-archive
  pruning parents were eligible. No search-performance or diversity superiority
  claim is supported by these three-epoch engineering runs.
- `reports/julia-pruning-fixture-20260915/`: separate selected previous-GA RPA
  fixture, parent loss 0.000738378434. All five deletion trials correctly rejected;
  50 pruning evaluations plus one setup evaluation. One deletion had no feasible
  inherited start; the other losses were 1.047883416, 0.013895389, 0.017552198,
  78.030699810. Original archive unmodified. The regression suite additionally
  verifies a real Julia/CVODE accepted deletion on a synthetic dispensable
  unobserved reaction, while protecting the fixed observed dynamics.

Final combined verification: **68 tests passed** in 38.48 s, including real Julia
and CVODE, nonblocking scheduling, shared budget protection, resume/cache behavior,
task-loss agreement, reaction masks, exact per-topology caps, isolated start
failures, pruning correctness/tolerance, and existing stream/GA regressions.
All 214 frozen scientific source hashes remain unchanged. Reports and figures are
local; CPU-only pilots used no GPUs. See `RL4CRN/search_streams/JULIA_PILOT.md`
for commands, accounting definitions, supported objectives, and limitations.

Post-run artifact audit independently reconciled the 26 job files against all
1,032 charged joint attempts, checked bounded rates and topology-distinct
admissions, and confirmed extension counts 1/8/8 across submission epochs.
No job exceeded either per-phase cap; the terminal queue/reservation ledger was
empty. The generated figures were inspected and their terminal drain boundary
made explicit rather than presenting it as an extra search epoch.

### 2026-09-15: Full RPA Julia-Stream Evaluation Launched

User requested a real test after the three-epoch verification. Added
`run_julia_rpa_evaluation.py` to launch one seed-0, 100-epoch, batch-1023 run and
then render its report and a saved-endpoint comparison. Existing completed
evolution-fit hybrid v2 and bounded standalone GA controls are read-only inputs;
none is rerun or modified. Five focused controller/configuration/report tests passed.

The controller inherits the corrected hybrid's exact task definition, policy
size 1024, source proportions, archive/imitation settings, bounds and threshold
0.016491168958935352. The latter is stricter than the smoke threshold 0.05.
Only the nominal fit-evaluation field (24) and overall attempt cap (214793,
the completed reference's actual expenditure) change within ExplorationConfig.
The Julia backend, admission settings and worker allocation are explicit:
32 search processes, eight Julia fitters, eight independent canonical validators,
queue capacity 32, maximum 32 fitted topologies per epoch, default 24+24 recipe
and measured 20% fitting-time admission target. CPU only, no LLM calls.

This matches the hybrid's seed and 100-epoch search horizon, not exact actual
work. Joint loss/gradient attempts and state-only evaluations differ in cost;
unused credit is not force-spent. GA ran until its 214793-evaluation cap, not for
100 generations, and retains only 30 archive entries. The automatic comparison
therefore reports both all qualifying discovered structures and retained counts,
actual attempts and runtime, with explicit single-seed/compute caveats. Separate
post-pruning does not enter search endpoints.

Launch (from repository root):

```bash
tmux new-session -d -s julia-rpa-100epoch-20260915 \
  -c /local0/rossin/git/GenAI-Net \
  'env MPLCONFIGDIR=/local0/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m comparisons.search_exploration.run_julia_rpa_evaluation --output comparisons/search_exploration/reports/julia-rpa-100epoch-20260915 --project /local0/rossin/git/SynthEvo/SynthEvo'
```

Progress: `reports/julia-rpa-100epoch-20260915/progress.json`, refreshed every 15 s;
child metrics/status/database: `run/`; console output: `run.log`. End-of-run
artifacts: `REPORT.md`, `comparison.json`, and `run/REPORT.md` with plots. No
automatic retry or replacement after failure. A new asynchronous question asked
for this run's Comet upload authorization; absent an answer at launch, logging is
local only. No credentials or external uploads are included in the launch.

Launch verification: eight Julia workers ready; first two epochs completed with
best canonical loss 0.1979727033, one fitted result merged, nine jobs pending at
the second boundary. Known initial-start failures remain recorded; no fatal
error or controller failure occurred. This is an early status, not a result claim.

### 2026-09-15: Separate Numerical Warnings From Campaign-Aborting Disagreements

The full RPA test stopped after 92 completed search epochs at 15:08 local time.
The strict agreement gate rejected Julia loss 1.0405492305529764 versus complete
canonical CVODE loss 1.0414303766810842 (absolute gap 0.0008811461, relative gap
about 0.0846%). Both are far above the qualification threshold 0.01649116896.
The user correctly noted that this small discrepancy should not abort a campaign.
This observation alone does not diagnose the numerical source of the difference.

Kept the former `rtol=2e-4, atol=2e-6` gate as a diagnostic warning, recording
both losses, absolute/relative discrepancy, bounds, job identity and epoch in
the event ledger and job JSON. Canonical loss alone remains authoritative for
archive insertion and qualification. Added a separate conservative abort guard
at `abs_difference > max(1e-5, 0.01 * max(losses))`; larger differences still
fail visibly. Nonfinite/negative losses remain errors and incomplete canonical
simulations remain rejected. No solver tolerance, rate bound or task loss changed.

The policy is frozen explicitly in new manifests. Fatal job JSON now also
preserves both scores and diagnostics instead of only the Julia phases/error.
The failed campaign, its source archive and partial results are unchanged, and
no restart or new simulations were performed in this change.

Focused verification: 31 tests passed, five real-Julia tests skipped because
this change was tested without starting Julia. New tests reproduce the observed
warning, verify actual pool acceptance uses the canonical score, retain fatal
large/invalid-loss behavior, reject incomplete trajectories and verify saved
failure diagnostics. Existing nonblocking stream/controller tests passed; all
214 frozen source hashes and whitespace checks passed.

### 2026-09-15: User-Requested Full RPA Restart

Restarted from seed 0 at approximately 15:28 local time, in fresh tmux session
`julia-rpa-100epoch-restart-20260915`. New output root:
`reports/julia-rpa-100epoch-restart-20260915/`. The interrupted 92-epoch run,
its manifest, source snapshot and databases are untouched; this is a fresh run,
not an asserted exact asynchronous checkpoint resume.

Same 100 epochs, batch 1023, seed 0, NN size 1024, canonical RPA task,
qualification threshold 0.016491168958935352, 214793 attempt cap, 32 search
workers, eight Julia fitters, bounded asynchronous recipe and separate terminal
pruning. The new manifest explicitly freezes warning tolerances 2e-4/2e-6,
abort tolerances 0.01/1e-5 and canonical-loss authority. No other search settings
changed. No LLM calls or Comet uploads; upload authorization remains unanswered.

Command is the preceding full-evaluation launch with session/output names
changed to `julia-rpa-100epoch-restart-20260915`. Confirmed controller alive,
new manifest contains the intended policy and worker allocation, and no failure
artifacts existed during initialization. Eleven targeted policy/controller
regressions passed (14 unrelated tests deselected).

### 2026-09-15: Full RPA Restart Completed; Report-Only Recovery

All 100 epochs and 1796 fitting jobs completed in 1149.35 s (19.16 min).
Canonical best loss 0.001043882054; 144 qualifying archived/discovered topologies,
58 qualifying descriptor regions. 1601 fitting jobs improved their inherited
candidate; 106 initially incomplete inherited candidates became canonically
complete. Budget: 135993 attempts = 74609 canonical + 61384 joint, with 547480
Julia scenario solves. Fitting consumed 19.98% of measured worker time; no
pending jobs or reserved evaluations remained. Five pruning parents were tested
with 279 separate evaluations; two single-reaction deletions were accepted.
Original search archives remain unchanged.

Automatic reporting, not search, then failed because `outcome.best` can be null
for a legitimately unsuccessful fitting job. Fixed the renderer's rescue count
to handle that case, added a regression test, and added `--report-only` to the
controller. Rebuilt reports and comparison from saved artifacts without any
search/evaluator/model calls. `report_recovery.json` preserves the prior failure
status and recovery operation; the original `controller_failed.json` remains.
Live progress now correctly reports completion. Six focused controller/report
tests passed. Figure labels were adjusted to avoid overlap at the drain boundary.

Saved-control comparison (descriptive single seed, not equal actual compute):
Julia hybrid loss 0.00104388 / 144 qualifying discoveries / 19.2 min;
evolution-fit hybrid 0.00254877 / 176 / 28.1 min;
standalone GA 0.000723983 / 126 / 56.8 min. Controls each used 214793 state-only
attempts versus the Julia arm's smaller mixed-attempt count; worker allocations
and fitting policies differ. GA retains only 30 live entries, so discovered
counts, not retained counts, are the relevant cross-archive comparison.

### 2026-09-15: Frozen 20-Seed RPA Workflow Comparison

User requested statistically meaningful evidence beyond the seed-0 engineering
comparison. Added RPA_STATISTICAL_PLAN.md, rpa_statistical_campaign.py and
analyze_rpa_statistical.py. Freeze 20 new matched seed identifiers (1..20) in
three arms: Julia-fit hybrid, evolution-fit hybrid, bounded standalone GA.
The seed-0 results remain unchanged and outside the new inferential cohort.

Hybrids keep 100 epochs, batch 1023, intermediate size 1024, threshold
0.016491168958935352, bounds [0.001,100], existing learning and fitting recipes.
All arms receive a new common 214793-attempt ceiling; the hybrid protected
search reserve is 102300, so optimizer credit cannot exceed 112493. This differs
from the original evolution-fit cap and is explicitly documented. Unequal
actual work and gradient-vs-canonical cost prevent equal-compute or pure
optimizer-effect claims. Separate post-search pruning is disabled.

Primary endpoint counts all qualifying discovered topology hashes in the design
ledger, not unequal live archive sizes. Two Julia-v-control contrasts on yield
and loss form a four-test Holm family. Report exact sign-permutation Wilcoxon
tests (Pratt zeros), paired W/T/L, paired bootstrap effect/ratio intervals.
No interim significance tests; inference requires all 60 runs complete and
validated. Failures and frozen-source changes hold dispatch, without automatic
retries or exclusion. Progress and local CSV/JSON/Markdown/PNG/PDF refreshes are
included. One 48-worker hybrid lane overlaps one 16-worker GA lane.

Added optional --seed and --no-comet to the standalone GA runner; original
defaults and mutation/search behavior are preserved. This campaign remains
local-only; no LLM or GPU use and no new Comet authorization assumed.
Focused verification: 46 passed, 5 Julia-environment-dependent tests skipped;
all 214 legacy scientific source hashes match. Host Julia checks follow before
launch. No legacy scientific modules, existing outputs or manuscript edited.

Host Julia verification completed: all 20 tests in test_julia_search_stream.py
passed, including the five previously skipped real Julia cases. Prepared and
launched `reports/rpa-statistical-20seed-20260915` in tmux session
`rpa-statistical-20seed-20260915`. Plan hashes cover 55 new-namespace source/plan
files and 47 config/reference/Julia inputs, in addition to the 214 preserved
legacy files. Verified the first Julia and GA manifests: seed 1, common cap,
correct rate bounds, explicit Julia recipe. Controller shows two running jobs,
58 queued, no failures, 64 allocated worker slots. Local analysis artifacts are
present. Exact prepare/launch and read-only refresh commands are in the protocol.

Startup verification progressed to Julia epoch 4 and GA batch 9, with canonical
evaluations and Julia fitting merges recorded and no failure artifacts. All
five Julia-dependent cases were exercised successfully (51 distinct focused
test cases passed across the runs). A synthetic 20-pair exhaustive statistical
check also completed; these synthetic values are not experiment results and
were not written into campaign data. Exact endpoint inference may take several
minutes after the last run. Old benchmark data were not reevaluated.

### 2026-09-16: Explicit Count-Budget Recovery Without Scientific Timeouts

User requested campaign continuation without wall-clock evaluation limits.
The campaign held after ga-seed02 exceeded 30 seconds on one evaluation; its
first attempt charged 3069 evaluations (3068 completed). Four other runs fully
completed. Audit found zero evaluation-failure events in those four, no fitting
deadline terminations, and maximum fitting-job times below ten seconds, so
they are retained rather than rerun.

Opt-in None deadlines added to exploration config, canonical evaluation pools,
evolution fitting, Julia client/topology fitting/extensions, and coalesced cache
waits. Existing finite defaults remain. Returning failed worker slots ensures
queued calls fail visibly on a dead connection instead of blocking forever for
a vanished slot. Numerical errors, solver controls, rate bounds, attempt limits,
and explicit cancellation still apply. Startup/shutdown health checks remain.
Julia's 20% measured-time admission scheduling is unchanged; it no longer has
per-job clock termination. This still compares common ceilings, not equal work.

Added resume_rpa_statistical.py and explicit --amend-no-timeouts controller
preparation. Original plan/configs/progress snapshot/source archive/failed GA
attempt remain preserved. active_plan.json carries completed jobs forward and
dispatches queued jobs with null deadlines to distinct output directories.
GA seed 2 restarts from its original seed because no GA checkpoint exists;
its first attempt's charged work is disclosed separately, not hidden, reused as
a result, or deducted from the new attempt's full 214793 ceiling. Analysis labels
deadline-policy versions and retry overhead. No automatic retry or cherry-picked
replacement of completed poor results; future genuine failures still hold.

Verification before host Julia checks: 74 focused tests passed, six real-Julia
cases skipped; all 214 legacy scientific source hashes match. No manuscript,
legacy results, scientific loss function or budget ceiling changed.

Host Julia checks passed: 41 tests including real no-timeout fitting and retained
finite-timeout error behavior. Prepared the amendment and resumed tmux session
`rpa-statistical-20seed-20260915` via `--execute-prepared`. Controller verified
running with four retained completions, Julia seed 2 and GA seed 2 active,
54 queued, and 64 allocated workers. Local analysis refreshed successfully and
explicitly discloses the amendment and 3069 failed-attempt overhead. Active plan
and config hashes matched at dispatch; original plan and attempts remain intact.

### 2026-09-16: Read-Only Qualification-Threshold Sensitivity

Added standalone plot_rpa_thresholds.py and three focused tests. It reads the
minimum complete recorded loss per topology from the full designs ledger,
including evicted designs and those outside the frozen qualification flag.
Counts use loss strictly below each threshold. Current completed matched seed
blocks are plotted separately from seed-0 development runs; incomplete or
unmatched seeds are not pooled. PNG/PDF, CSV curves and JSON/Markdown summaries
are under analysis/threshold_sensitivity. This is post-hoc descriptive analysis,
not a changed primary threshold, formal diversity claim or significance test.

Current seed 1 counts at threshold 0.005: Julia 10, evolution-fit 87, GA 0;
at the frozen 0.016491168958935352: 119/203/0. Seed-0 development counts at the
frozen threshold: 144/176/126. Figures use a log threshold axis and a count axis
linear from 0 to 10 and logarithmic above 10, explicitly labeled to retain zeros.
No simulations rerun, campaign implementation/source hashes untouched. Three
tests passed; the rendered figure was inspected and its common vertical limit
adjusted to include both panels without clipping development GA counts.

### 2026-09-16: Increase Campaign Concurrency Without Restarting Active Runs

User requested more overlapping runs. Host has 64 physical cores / 128 threads;
203 GiB RAM available and 532 GiB free on the output filesystem. Three-second
CPU sample was 37.7% of logical capacity. Added explicit two-hybrid/two-GA lane
limits (128 worker slots), preserving all per-run workers, seeds, budgets,
no-timeout settings and scientific source modules.

Original controller acknowledges HOLD and drains its existing children.
scale_rpa_statistical.py adopts their PIDs by reading the original parent's
reported exit codes, rather than guessing completion from files or interrupting
workers. It owns remaining queued jobs. New state/report paths avoid concurrent
writers: progress-concurrent.json and analysis-concurrent/REPORT.md, selected by
active_plan.json. Concurrency source/plan/progress snapshots and disclosure are
kept in amendments/concurrency. Runtime/asynchronous schedule effects are not
claimed absent. Existing failure holds and scientific source checks remain.
Dispatch records include the worker/lane limits. No new LLM calls or uploads.

Handoff verified: Julia seed 5 completed normally during preparation. The new
controller adopted the still-running GA seed 4 with the same PID, then launched
evolution seeds 5 and 6 plus GA seed 5. Live state reports 12 complete, four
running, 44 queued, no new failures and 128 allocated worker slots. New tmux
session: rpa-statistical-concurrent-20260916. Current report generated at
analysis-concurrent/REPORT.md. Seventeen queue/adoption/statistics/threshold
tests passed; all 214 preserved scientific source hashes still match.

### 2026-09-16: Finite Julia/Canonical Disagreement Becomes Warning-Only

User explicitly requested continuation without stopping on score disagreements.
Added an opt-in JuliaRecipe loss_disagreement_policy="warn". Every finite,
nonnegative pair now yields an agreement or warning under that policy, including
differences above the historical 1% bound. Large differences keep a separate flag
and counter; both scores and differences remain in job diagnostics/events. Only
the canonical score enters the accepted candidate. Existing default abort policy
is preserved for older callers. Nonfinite/negative scores, incomplete canonical
simulations and actual worker/program errors are not silently accepted.

Added resume_rpa_warnings.py to preserve the concurrent plan/progress/source
snapshot, retain all completed outcomes, and restart only interrupted Julia seed
6 in a new output folder. All queued Julia jobs receive the warning recipe;
GA and evolution commands, budgets, four-run concurrency and no-timeout settings
remain unchanged. Aborted seed-6 work is retained as additional recovery overhead.
The report identifies the new policy and mixed-policy cohort explicitly.

Verification: 39 focused tests passed with six Julia-dependent skips; all 23
Julia stream tests then passed on the host, including warning-mode canonical
acceptance and retained strict-mode/error handling. All 214 legacy hashes match.
Prepared amendment/counter snapshots and restarted tmux session
rpa-statistical-concurrent-20260916. Eighteen completions retained, 42 jobs queued
at restart; interrupted Julia seed 6 starts in julia-seed06-canonical-warnings.
Its previous attempt charged 120140 canonical/joint attempts, separately retained
alongside the earlier GA recovery overhead. No scientific wall-clock limits or
evaluation ceilings were changed by this agreement-policy amendment.

## 2026-09-16: Disk-full relocation and explicit recovery

User authorized moving the campaign to /local1/rossin and recovering. Confirmed
all four active jobs and the controller had exited; logs reported SQLite disk
full and ENOSPC, not scientific failures. Copied 90,619,869,621 bytes (28,362
files) with rsync, then checksum-verified the complete destination with an empty
`rsync -acn --itemize-changes` diff before removing verified source copies.
The campaign now lives at
`/local1/rossin/search-exploration/rpa-statistical-20seed-20260915`; the original
repository path is a symlink. Approximately 85 GiB was released on /local0.
No scientific artifacts, failed attempts or unrelated files were discarded.

Added recover_rpa_disk.py with explicit one-shot preparation/launch, both
controller locks, frozen-input checks, completed endpoint and SQLite integrity
validation, separate failed-attempt accounting and output collision guards.
All 26 completed runs passed validation and were retained. All four interrupted
databases passed integrity checks as well. Since exact asynchronous checkpoint
resumption is unsupported, GA seeds 9/10 and Julia/evolution seed 10 restart
with the same settings in new -disk-recovery folders. Their old attempts stay
untouched. Recorded disk-failed work is 364124 charged evaluations in total,
reported as a lower bound because final counter persistence can fail on a full
disk. Partial discoveries are not pooled or selected against rerun outcomes.

The recovery plan keeps all 60 scientific commands identical except output
paths; budgets, seeds, four-run concurrency and no-timeout/warning policies are
unchanged. Thirty never-started jobs remain in the queue. The report now includes
an explicit disk-recovery disclosure linked to the amendment and database audit.
No new Comet uploads or LLM calls. Source and plan snapshots are preserved.

Verification: 19 focused tests passed; all 214 frozen legacy source hashes
verified during preparation. Launched tmux session
`rpa-statistical-local1-20260916` with recover_rpa_disk --execute-prepared.
Live status remains progress-concurrent.json and analysis-concurrent/REPORT.md
through either the original symlink or the physical /local1 campaign directory.

## 2026-09-17: Completed-cohort RPA portfolio and mechanism atlas

Added analyze_rpa_mechanisms.py as a separate post-hoc analysis, without changing
the frozen campaign implementation or rerunning any evaluator/model. Output is
`/local1/rossin/search-exploration/rpa-statistical-20seed-20260915/analysis-mechanisms-20260917`.
The analysis requires all 60 validated endpoints and reproduces their best losses
and qualifying counts from read-only design ledgers. It uses only final completed
attempts, not interrupted runs or developmental seed 0. Extracted best-per-topology
records are cached against the active-plan hash. All figures have vector PDF and
PNG versions, captions, CSV/JSON backing data, and a combined PDF atlas/report.

Quality threshold curves, within-run pairwise Jaccard distances, exact cross-run
topology overlap, symmetric nearest-structure similarity and average-linkage
clustering cover 3239 distinct qualifying labelled structures. Structural tokens
retain stoichiometric multiplicity and input channels; fixed template subtraction
uses full structures, never reaction IDs alone. The exploratory 0.60 cut gives
85 families, with cuts 0.30/0.45/0.75 reported for sensitivity. Pairwise cells and
topologies are not treated as independent statistical replicates.

Added exact rarefaction to ten qualifying structures, with the plot restricted
to the ten matched seeds eligible in all three workflows. Expected family counts
are 6.97 / 6.87 / 4.61 for AdaGrad hybrid / evolution-fit hybrid / GA. Raw mean
family counts across all seeds are 23.8 / 21.2 / 8.55. These are descriptive,
post-hoc structural results, not new significance claims or mechanism counts.

All 60 best-network cached diagnostics were available. Functional panels separate
endpoint tracking, disturbance spread on the existing nine-input grid, and
all-state tail drift. Representative panels use cached trajectories and computed
mass-action terminal fluxes, not fresh ODE solves. Some structures maintain output
while an internal species accumulates; bounded-state RPA cannot be inferred from
low finite-horizon loss. A common exchange pair contributes
X2*(k51*X1-k80*X3) to output dynamics. A separate restricted polynomial-balance
test identifies d(X1+a X2)/dt = u1-g OUT, with explicit numerical tolerances;
an illustrative lowest-loss cached exact-balance candidate complements the six
most recurrent structural-family representatives. Neither diagnostic proves
stability, causal necessity, or robustness to unseen inputs.

Verification: 19 focused tests passed, including preservation of an unmodulated
reaction sharing a fixed reaction ID, order invariance/multiplicity, empty-set
handling, exact rarefaction versus enumeration, an antithetic balance fixture,
and hidden-state drift despite perfect output. Checked symmetric 60x60 matrices,
all 60 extracted endpoint counts and all representative cache availability.
Visually inspected representative, clustering, diversity plots and PDF summary.
Final atlas: 15 figures and 18 PDF pages. All seven representative reconstructions
match canonical reaction propensities at all nine cached terminal states to
1e-12 tolerance. All frozen campaign source/input/plan/legacy checks still pass.
The illustrative exact-balance GA seed-13 design has loss 0.000521979, gain
0.999989910, worst final relative output error 1.009e-5 and maximum all-state
tail change 9.027e-10 on the saved grid. This positive contrast is retained
alongside accumulation-based examples, not hidden by the hybrid comparison.

## 2026-09-17: Completed RPA kinetic-perturbation study

Added run_rpa_perturbations.py, analyze_rpa_perturbations.py and
audit_rpa_perturbation_failures.py. Output is isolated at
`/local1/rossin/search-exploration/rpa-rate-perturbations-20260917`.
No legacy/hybrid evaluator, campaign result, prompt or manuscript was modified.

Frozen study: all 60 current rank-one designs and 19 historical RL-only winners.
Historical RL-only seed 10 is preserved but explicitly excluded because its saved
winner contains the now-forbidden null reaction; no replacement winner is used.
Rates are indexed by reaction ID, with common antithetic log-uniform directions
and radii 1/5/10/25/50 percent. Thirty-two draws per radius/design, ten one-rate
tests per design, fresh baselines, and balance-preserving controls for 30 designs
give 18,309 evaluations. All use the unchanged canonical nine-scenario CVODE
task, 100 time units, 1000 points, tolerances 1e-8. Fixed template coefficients
remain unchanged. Separate diagnostic bounds admit perturbations beyond search
bounds without clipping. There is no refitting, LLM call, Comet upload or timeout.

Execution used 64 persistent spawned workers with single-threaded BLAS, then
finished. SQLite stores every result. Frozen hashes prevent silent changes;
resuming submits only missing jobs and never retries away saved failures.
All 79 baselines reproduced stored losses within 1% (maximum relative drift
approximately 9.94e-7). Failed/incomplete draws remain in every denominator and
are treated as infinite loss for ordering summaries, not given a chosen penalty.

At radius 10%, original-threshold retention among baseline-qualified winners is
15.3% historical RL-only, 15.0% AdaGrad hybrid, 11.1% evolution-fit hybrid and
8.2% GA. These conditional probabilities are distinct from absolute function:
across all eligible winners, disturbance-spread <5% pass fractions are
82.4%, 100%, 97.7% and 57.2%. Low baseline losses inflate loss ratios; plots also
show absolute degradation and baseline headroom. These are post-hoc descriptive
comparisons, not new between-method significance tests or equal-budget claims.

For the 30 exact-balance designs, median network-median loss inflation at radius
10% is about 18.8x when all rates vary versus 1.0026x under balance-preserving
changes; median worst-case relative final errors are 5.21% versus 0.0292%.
This control varies 1-4 rates instead of five and is explicitly not a fair
equal-dimensional robustness contest. It separates calibration/balance
sensitivity from disturbance rejection; no stability or RPA proof is asserted.

All 651 incomplete results were replayed solely for raw solver diagnostics with
16 workers, adding 651 evaluations. Every status and loss was reproduced; every
case hit the existing all-state ceiling of 10,000, with zero CVODE solver failures.
No original outcome was replaced. GA accounts for 94/640 incomplete all-rate
draws already at radius 1%; 65 of those originate from qualifying baselines.
Four of its affected winners have baseline terminal maxima around 9964-9993.
This is state-ceiling sensitivity, not evidence by itself of mathematical
divergence. Per-scenario raw statuses/messages and times are in failure_diagnostics.json.

Outputs: five figures with separate vector PDF/PNG exports, combined PDF,
Markdown interpretation/protocol, baseline audits, per-network sensitivities,
one-rate tests, paired balance controls, state headroom and full failure records.
Focused verification covers deterministic/order-invariant perturbations,
sensor-gain masking, immutable baselines, job uniqueness, failure denominators,
reevaluated-baseline qualification, infinite failure losses and seed bootstrap.
Final verification: 14 focused tests passed; all 66 perturbation input/source
hashes, its plan hash and the parent campaign's frozen checks passed. Visually
inspected plots and protocol page; the combined PDF contains six pages.

## 2026-09-17: Absolute perturbation survivors and atlas note

Added analyze_rpa_perturbation_counts.py and three focused regression tests.
It reads the frozen completed perturbation inventory without simulations and
separates retained qualification from new qualification of previously poor
baselines. It counts successful evaluations, saved winners passing any/half/all
draws, and average/minimum/maximum retained winner counts across directions.
Perturbation draws are not presented as distinct structures or independent runs.

Wrote PERTURBATION_FOLLOWUP.md beside RPA_PORTFOLIO_ATLAS.pdf, with links to the
separate perturbation report/PDF and new absolute_counts.csv/.json. At radius
10%, initially qualifying winners yield 44/288 historical RL, 96/640 AdaGrad,
71/640 evolution-fit and 34/416 GA successful tests: mean remaining counts
1.375, 3.000, 2.21875 and 1.0625 respectively. No initially qualifying winner
passes half the all-rate directions at this radius. Historical RL also has ten
successful tests from previously nonqualifying baselines, disclosed separately.
Balance-preserving all-32 survivors are 7/8 historical RL, 9/9 AdaGrad, 6/6
evolution-fit, 4/4 GA, conditional on qualifying exact-balance baselines.
The note explicitly rules out extrapolating these winner-retention rates to
all archived topologies. Eleven focused count/perturbation tests passed.

## 2026-09-17: Manuscript appendix on GA and parameter-fitting comparisons

Added the final manuscript Appendix M, Genetic Search and Parameter-Optimization
Comparisons, in sections/appendix_ga_optimization.tex. It separates the no-LLM
workflow experiment from the existing communication cohorts, including unequal
actual attempt counts, joint-gradient accounting, archive/fitting differences,
the primary four-test family, and post-hoc 15-test perturbation family.
It reports evolution-fit's larger topology yield alongside AdaGrad's better
nominal loss, and treats the perturbation tendency versus evolution-fit as
selection motivation rather than corrected significance. All GA perturbation
contrasts are reported, including failures and baseline-quality dependence.

generate_ga_optimizer_appendix.py produces three tables, two paper-sized vector
figures, JSON input hashes/results, and all 15 post-hoc contrasts in CSV. Exact
Pratt signed-rank p-values use integer-rank dynamic programming, checked against
SciPy exhaustive permutation tests on tied/zero fixtures. No simulations or
model calls were made, and no frozen campaign implementation was changed.
The build audit passed; 17 focused statistics/editorial tests passed. Rebuilt
and visually inspected pages 39-41. Total PDF length is 41 pages (previously
38), with no overfull boxes or unresolved references; main-text pagination
is unchanged. Existing underfull warnings are outside the new appendix.

For the next LLM extension, RPA has the validated AdaGrad recipe and 20-seed
no-LLM controls; Logic is the stronger follow-up for structural complementarity.
Inspection confirms the asynchronous AdaGrad runner currently has no Harness
option and hard-codes harness=False. The older exploration runner's Harness
support must not be mistaken for an enabled option in the new stream runner.
No LLM campaign was launched by this manuscript update.

## 2026-09-17: RPA Flash Harness readiness, not a campaign

Saved the approved evidence-guided RPA task prompt in a separate versioned file,
reusing the new workspace-loss system and Decider/Writer prompts unchanged.
Added check_rpa_flash_harness.py with explicit --live opt-in, ten-candidate
batching, canonical CVODE evaluation, independent-member validation and a
withheld initial HOF. It does not fit rates or train RL, cannot overwrite an
output directory, and never copies credentials into artifacts.

Offline preflight passed with Node v24.19.0 and the pinned DSH rc.8 runtime.
39 focused Python tests and one Node model-selection test passed. Runtime
credentials exist but validity is unverified. Automatic approval blocked the
live request as an external disclosure to DeepSeek; explicit payload/destination
approval was requested. No model call or Comet upload occurred, and the denied
command was not retried through another route. See RPA_FLASH_HARNESS_READINESS.md.
The asynchronous AdaGrad runner still needs its own Harness integration; the
older runner's communication tests do not prove that integration is in place.

## 2026-09-17: Full-duplex Flash integrated with asynchronous AdaGrad

Added isolated opt-in search_streams/harness.py, run_harness_streams.py and
rpa_flash_stream_campaign.py. The frozen no-LLM stream runner and fitting
implementation were not modified. New integration reuses the Decider/Writer
graph, canonical evaluator/cache, archive and background model executor. It
withholds initial HOF/EIL/exclusion/aggregate state, restores subsequent outward
communication, and admits LLM candidates to the existing fitting selection.
Incomplete initial trajectories may be fitted but are not successful archive
entries. Infrastructure evaluation failures propagate; invalid LLM payloads
remain recorded request outcomes. Fixed terminal LLM accounting in the opt-in
subclass and retained distinct training/terminal archive snapshots.

Live readiness succeeded after explicit approval: 10/10 valid, best loss
0.005954755380383372, no HOF exposure, two roles, 468.94 seconds. Seven final
adapter/readiness tests passed, plus broader stream/protocol tests and three
real persistent-Julia continuation/budget tests. No numerical wall-clock
timeouts were added. AdaGrad uses the established loss-warning policy with
canonical scores authoritative, matching the final reference configuration.

The first eight startups failed before search because of a stale inherited
Comet credential. Preserved diagnostics and redacted SDK credential echoes.
Verified the existing local Comet configuration and launched a fresh -r1 root,
with the controller removing only the stale credential environment override.
20 seeds, 100 epochs, 1023 proposals, total cap 214793; eight seeds concurrent
with six search/two Julia workers each. Eight Comet experiments are live and
all eight first workspaces passed withholding audits. All eight reached epoch
1 with model requests pending. The initial utilization sample was 48.3 core
equivalents; concurrency allocation and its comparison caveat are recorded.

Active root: /local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1
tmux: rpa-flash-adagrad-20260917. See RPA_FLASH_ADAGRAD_CAMPAIGN.md for workflow,
failure history, monitoring paths and comparison limitations. No paper figures
or previous experimental results were changed in this integration.

## 2026-09-17: Directional AdaGrad/Flash Arms and Reused Independent Union

Added isolated directional_harness.py, run_directional_streams.py,
rpa_directional_stream_campaign.py and analyze_rpa_directional_streams.py.
Insertion-only hides all outward search state; return-only bypasses both archive
insertion and incomplete-proposal fitting, with fail-closed checks against search
contamination. Both preserve own-proposal evaluator feedback, Decider/Writer,
request-zero withholding, canonical validation, budget accounting and Comet.
Direct supplied-rate proposals have a separate per-member JSONL ledger.

Launched 20 seeds per directional arm, 100 epochs, ten seeds concurrent. The
previous full-duplex cohort is now 20/20 complete. No old source hashes changed.
The user-requested no-comparison reuse pools historical matched no-LLM AdaGrad/RL
designs with insertion-only raw LLM proposals, without new model calls. Reports
explicitly label this post-hoc independent aggregation, not matched asynchronous
no-communication or causal replay. Context, identity and conservative total-cap
audits gate each reused union. No descendants or HOF-conditioned banks are pooled.

43 focused tests passed. All ten first live workspaces passed initial context
isolation checks; ten Comet experiments are live. New and old frozen scientific
source hashes match. No manuscript or historical result was modified. See
RPA_DIRECTIONAL_ADAGRAD_CAMPAIGN.md for settings, commands and limitations.

## 2026-09-17: Completed Full-Duplex versus No-LLM AdaGrad/RL Analysis

Added analyze_rpa_flash_adagrad.py and focused regression tests. Read all 40
completed endpoints without simulations, model calls or database writes. Checked
reaction/rate constraints, historical control audit consistency, shared task
builders from frozen source archives, settings and drained work. Configuration
differences are recorded rather than silently treated as a pure ablation.

Twenty paired seeds: lower best loss with Flash in 20/20 (median paired ratio
0.2532, Holm p=3.815e-6); greater qualifying-topology yield in 15/20 (mean gain
20.35, bootstrap CI 6.25-33.90, Holm p=0.00828). Mean counts 152.55 versus 172.90.
Threshold sensitivity shows stronger count gains at stricter quality criteria,
but reversal at loose thresholds. No universal diversity or return-channel
benefit is inferred. Numerical expenditure differs, and LLM inference is extra.

Generated and visually inspected three PNG/vector-PDF figures and a combined
three-page PDF in the full-duplex root's analysis-vs-adagrad-20260917 folder.
Reports distinguish terminal draining, retained archives, all discoveries,
sample SD, pointwise confidence intervals and corrected endpoint tests. Eight
focused tests passed. See reports/RPA_FLASH_ADAGRAD_COMPARISON.md. No manuscript,
historical results or running experiment code was changed.

## 2026-09-18: Recover Provider-Balance-Affected Directional Records

Audited all 435 directional request histories and saved error files: 199 failures
were insufficient provider balance, affecting six insertion-only and seven
return-only records. The search processes had completed but those endpoints are
not clean communication interventions. Kept all original artifacts; added a
recovery pointer/warning without rewriting the original results.

Added recover_rpa_directional_balance.py and regression tests. Recovery changes
only output paths in the original affected commands, checks frozen source/input
hashes, preserves the 27 unaffected endpoints, and restarts 13 same-seed records
without selecting the better result. Original discarded attempts' 1,775,346
numerical calls remain explicit overhead. Empty/invalid scientific payloads are
not automatically replaced. Added one-minute protected balance checks and
request/stderr guards; renewed balance failure holds dispatch and terminates
owned active recovery process groups. No automatic retries or scientific code
changes were introduced. Comet integration is unchanged.

Verified available balance before launch and started a separate tmux controller
with ten active replacements and three queued. All ten created new Comet records.
The audited 12-seed unaffected subset has no clear directional/full-duplex lead;
the reused independent union currently has the largest mean qualifying count.
Reports label this descriptive and preserve the terminal-aggregation and
unmatched-schedule caveats. See RPA_DIRECTIONAL_BALANCE_RECOVERY.md for exact
paths, settings, preliminary values and monitoring details.

## 2026-09-18: LLM Provenance/Interaction and Winner Perturbations

Added analyze_rpa_llm_provenance.py, run_rpa_llm_robustness.py and
analyze_rpa_llm_functional_robustness.py, with provenance/cache regression tests.
The emitter/fitter of winning rates is distinguished from first structural
occurrence, explicit mutation origin, and request-time HOF exposure. Reconstructed
all 20 full-duplex endpoint losses exactly from event histories. Twelve winning
structures first appeared in RL samples, although all winning rate vectors come
through LLM proposals/fits. Four winning structures were exposed as RL-labelled
HOF entries before their first LLM proposal. These observations are not treated
as causal proof of return-channel superiority.

The qualifying set averages 172.90 structures, 136.95 absent from every direct
LLM proposal. First qualifications are mainly RL-emitted; explicit LLM-lineage
mutation contributes 1.35 per seed. The report preserves policy-ancestry and
temporal/copying limitations rather than equating emitter with invention.

Executed the established winner perturbation protocol for 76 logical winners
(66 exact distinct vectors), with eight workers and Comet logging. Reused 4,860
old evaluations and ran 14,266 new evaluations, no model calls or refitting.
All 19,126 outcomes persisted, including 43 failed/incomplete cases. Baselines
were valid without >1% drift. At 10% radius, qualification survival is 20.31%
with Flash versus 15.00% without; the paired-bootstrap interval for the gain
includes zero. Disturbance rejection survives in 99.53% versus 100% of draws,
showing why setpoint/loss sensitivity must not be equated with lost adaptation.

Generated and visually checked three provenance and two robustness figure pages,
with reports, CSV/JSON and audit records under rpa-llm-interaction-20260918 on
/local1. Sixteen focused tests passed. Recovery experiments continue unchanged;
directional robustness is explicitly the frozen 12-seed unaffected subset.
See reports/RPA_LLM_INTERACTION.md. The manuscript was not edited.

## 2026-09-18: Completed Communication Topology-Family Atlas

Added read-only analyze_rpa_communication_families.py and four regression tests.
All 100 arm/seed endpoints reproduce the audited summaries using the reconciled
directional output paths, not the balance-affected original attempts. Return-only
and reused independent-union portfolios retain isolated direct LLM proposals.
Clustering uses full input-aware structural records, exact fixed-template
subtraction, rate/order-invariant identity, average-linkage Jaccard distances,
cut sensitivity, and exact rarefaction to 10/50 topologies per seed.

The pooled cohort contains 1,453 qualifying topologies and 88 families at cut
0.60. Full duplex averages 29.65 families versus 23.70 without LLM, and 19.51
versus 16.45 after rarefaction to 50. Insertion only averages 19.97. No claim of
full-duplex superiority is made. AIF-like exact-balance fractions shift from
31.08% without LLM to 80.93% full duplex and 87.76% insertion only. Across-seed
exact overlap also increases, distinguishing local breadth from global novelty.
These motif screens are explicitly not proofs of stability or biological function.

Six final PDF/PNG pages plus a combined atlas, input hashes, extraction snapshot,
memberships, medoids, per-seed metrics and exploratory bootstrap intervals are
under rpa-llm-interaction-20260918/topology-families on /local1. See
reports/RPA_COMMUNICATION_FAMILIES.md. Fifteen focused tests passed. No new
simulations or model calls; previous experiments, provenance and perturbation
artifacts and manuscript sources remain unchanged.

## 2026-09-18: Independent-Union Source Decomposition

Added analyze_rpa_independent_sources.py and three regression tests. Reused the
audited 20-seed atlas extraction and complete direct supplied-rate proposal banks;
re-audited request-time withholding and reconstructed every union endpoint.
Frozen global family labels are reused, without source-specific reclustering.
Event-time control provenance separates RL proposals, their AdaGrad fits, and
mutation/random proposals/fits at first qualification.

Search contributes 152.55 qualifying topologies per seed, direct LLM 37.05,
and their overlap is 3.05. Exclusive additions are 149.50 search and 34.00 LLM;
exclusive structural families are 20.65 and 8.75. LLM wins endpoint loss in
all 20 seeds, with median paired loss ratio 0.23484. Of the search qualifications,
139.40 arise at RL proposal rates and 11.65 after AdaGrad fitting, versus 1.50
from other search emitters/fits. The report distinguishes terminal complementarity
from interaction, causal attribution, and equal-budget inference.

Two PDF/PNG pages plus source summaries, topology membership, input hashes,
threshold curves and search provenance are under
rpa-llm-interaction-20260918/independent-union-sources on /local1. Twelve focused
tests passed; figures visually checked. No new evaluations or model calls,
source database mutations, manuscript edits or changes to existing figures.

## 2026-09-18: Independent-Union Loss Histories

Added plot_rpa_independent_loss_time.py and three focused timing tests. Source
losses use observed metric snapshots; direct LLM losses use only the corresponding
received-record prefix, excluding incomplete scores. No backdating to request
launch or substitution of the donor search archive loss. Epoch 0 means completed
training epoch 1; terminal draining is a separate T point. Missing sources remain
infinite before upper-median and quartile aggregation.

The figure includes epoch-aligned trajectories, descriptive own-clock elapsed
trajectories, availability and loss leadership. A dashed virtual union is clearly
marked as a reconstruction of separate sources, not an executed parallel replay
or controlled wall-time comparison. All 40 source endpoints reproduce the audited
components. One PDF/PNG, snapshot/aggregate CSVs, arrays, input hashes and report
are under rpa-llm-interaction-20260918/independent-union-loss-time. Three tests
passed and the figure was visually inspected. No new simulations or model calls.

## 2026-09-18: RPA-Style Parity-12 Prompt Draft

Added prompts/parity12-evidence-exploitation-v2.md for review, retaining the
existing workspace-loss-v1 system and Decider/Writer templates. Adapted RPA's
precise weighted-mean objective, transient/stability caveats, source-neutral
parent selection, optional cached evidence, initial-HOF withholding, soft 6/4
proposal policy and limited AdaGrad fitting guidance. Parity targets explicitly
require OUT near 0/1 across all sixteen rows, with twelve added reactions and
production rate bounds [0.001,100]. No RPA-specific target or mechanism is copied.

Three focused tests check the prompt contract, truth table and numerical loss
weighting. Historical prompts/configurations and release approvals are untouched;
the file is not wired into a runner and no campaign is launched. See
reports/PARITY12_PROMPT_V2_REVIEW.md for integration requirements.

## 2026-09-18: Approved Parity LLM-Only Sequential Pilot

Added run_parity12_llm_pilot.py and four focused tests; together with prompt tests,
seven passed. Prepared and launched one sequence of twenty Flash/Harness request
rounds, ten candidate slots each, in tmux parity12-llm-only-20round. Task config
is a new copy with production bounds; original parity configs/prompts are unchanged.
Canonical evaluation uses the existing stream-compatible evaluator, a persistent
process, exact cache, and no numerical timeouts. No RL training or AdaGrad fitting.
Per-member validation is retained; malformed rounds are recorded without replacement,
and evaluator/provider infrastructure errors stop visibly. HOF, SIL and exclusion
state are withheld throughout; only standard own-proposal memory crosses rounds.

Comet is connected in genai-net-exploration. The first live workspace is empty of
HOF and previous self-feedback. Account-wide initial balance is USD 38.70, with
five-round snapshots configured. Sources/prompt/task are hash-frozen in a fresh
/local1 output directory. See reports/PARITY12_LLM_ONLY_PILOT.md for artifacts,
protocol, live status and Comet link. This is one sequential pilot, not twenty
replicated seeds; no completion claim is made before status.json confirms it.

## 2026-09-18: Four-Arm Draft and Harder-Task Screening

Started isolated parity four-arm infrastructure, then deferred dispatch when the
user requested investigation of a harder first-batch task. No new cohort launched;
draft/preflight status is explicit in reports/PARITY12_FOUR_ARM_PREPARATION.md.
Existing RPA/pilot runners and artifacts are unchanged.

Added audit_first_batch_difficulty.py: read-only queries of 160 completed audited
contract-v2 databases, verifying Flash identity and empty request-zero HOFs.
Compared twenty initial ten-proposal batches per task against the existing RL
endpoint median, and retained all later direct-LLM losses separately. This is a
post-hoc difficulty screen, not a functional success test or evidence that return
communication works. The new prompt and AdaGrad need separate validation.
Flagged the existing zero-weight mean term in oscillator_mean; no evaluator or
historical results changed. No new model calls or simulations for this screen.

## 2026-09-18: Self-LLM Request Curves

Added plot_self_llm_request_curves.py for the audited historical Flash
no-communication cohorts (RPA, Logic, classifier; twenty seeds each) and the
separate current parity pilot. Plots use only direct proposal losses, never RL
archive entries or fitted descendants. Every saved context is checked for
withheld cross-source information; actual self-feedback availability is exported
because asynchronous launches can precede earlier outcomes. Request-order
prefixes are not presented as observed wall-clock curves. Median/IQR bands are
descriptive, not significance intervals; the RL reference is not functional success.

Saved PDF/PNG figures, a two-page PDF report, CSV data and JSON provenance under
reports/self-llm-request-curves-20260918. The parity timeout is visible, without
inventing results for unfinished rounds. Dose-response and oscillator self-only
cohorts are absent from the audited registry and are not inferred from full-duplex
runs. No new model calls, simulations, or manuscript modifications.

The user then requested arrival-time alignment. Added plot_self_llm_arrivals.py
and reports/self-llm-arrival-time-20260918. Historical timestamps are host batch
collection relative to run initialization; parity uses logged canonical completion
relative to first workspace creation. Bests are recomputed in arrival order, with
infinite loss until a seed has a result. Unavailable seeds are not dropped from
order-statistic bands. Late arrivals and failed batches cannot backdate improvement.
Timings are descriptive and retain asynchronous scheduling/host-load effects.

## 2026-09-18: Original Five-Reaction Logic Prompt Review

Added prompts/logic5-evidence-exploitation-v2.md as a separate task prompt for
review, based on the approved evidence-guided parity/RPA instruction style.
Retains the workspace-loss-v1 shared system and Decider/Writer protocol unchanged.
Specifies the original adjacent-pair Boolean function, five species, four fixed
input reactions plus exactly five added reactions, canonical CVODE weighted
trajectory L1, production bounds [.001,100], initial-HOF withholding, optional
cached evidence, soft six/four refinement/exploration allocation, and host-only
evaluation/fitting. Historical prompts/configurations and runs are unchanged.
No campaign launched or prompt hashes in the manuscript regenerated. A future
runner must explicitly copy the legacy config with production rate bounds rather
than inherit its older [.1,50] interval. Focused tests check the live evaluator
mask/species/bounds, all truth-table rows and weighted-mean normalization.

## 2026-09-18: Approved Logic-5 Four-Arm Production Campaign

Added isolated run_logic5_four_arm.py, logic5_four_arm_campaign.py and
preflight_logic5_four_arm.py; reused the opt-in four_arm_harness adapter without
changing frozen RPA or parity code. Added four-arm routing and campaign-plan
tests. 32 focused checks passed, along with real Julia/CVODE preflight at original
and fitted rates (maximum observed discrepancy about 3.2e-8).

Prepared an immutable 80-run plan on /local1 and launched tmux logic5-four-arm:
twenty seeds per arm, 100 epochs, batch1023, cap214793, NN1024, ten concurrent
six-search/two-fit workers, Comet for every job. Initial ten request contexts were
verified HOF-empty. Numerical timeouts remain disabled; Harness roles receive
3600 seconds instead of900, and ordinary request failures do not stop numerical
search. Provider credit and source/disk integrity are monitored rather than
masking failures. Full settings, commands and caveats are documented in
reports/LOGIC5_FOUR_ARM_CAMPAIGN.md. No old run, prompt, manuscript or threshold
artifact was overwritten; no completion or scientific-effect claim is made.

## 2026-09-18: Live Logic-5 Scaling to Twelve Jobs

User approved increasing concurrency after observing CPU headroom. Added
scale_logic5_campaign.py using the existing RPA adoption protocol: original
controller acknowledges HOLD and reaps its own ten workers; the new controller
owns the remaining queue and counts all active jobs against twelve. No original
worker was interrupted or restarted. The two added jobs are full/llm-to-rl
seed03, making all four arms active for seeds01-03. All twelve have Comet records;
both new request-zero contexts have empty HOFs. Original process creation times
and all frozen scientific source/input/plan hashes were verified unchanged.

New session: logic5-four-arm-scaled. Canonical live state is now
progress-concurrent.json; progress.json is the original draining controller's
view. HOLD_CONCURRENT controls new dispatch. Do not remove original HOLD or
restart either controller. Immutable plan.json still records the initial ten;
amendments/concurrency12/amendment.json records the effective twelve, original
process identities, source digests and unchanged scientific settings. Failure,
credit, disk and hash checks remain enabled; scientific timeouts remain disabled.

28 focused tests passed, including real-exit-code adoption, scientific identity
reconciliation, combined concurrency and failed-adoptee dispatch holds. Worker
counts and the global eight-role Harness limit did not change. A thirty-second
baseline measured 62.6% logical CPU busy and 245.6 budgeted evaluations/second;
scaling-baseline.json and scaling-after.json retain raw counters and resource
measurements. Short windows are phase-dependent, not a controlled speed test.
Additional contention can affect asynchronous arrival/admission timing, and this
amendment explicitly retains that caveat.

First sixty-second post-scale window: 66.2% logical CPU busy, 140.8 GiB available
RAM, 182.9 budgeted evaluations/second. This is lower throughput than the short
baseline, despite higher CPU use, so no speedup is claimed. New workers were
still in their first epoch (search counts advanced 324->888 and 305->850 during
the window); existing workers reached epoch7. Twelve active, 68 queued, zero
failed. Sustained, phase-matched measurements are needed before increasing again.

## 2026-09-19: Explicit Logic Balance Recovery

Following user authorization and restored provider balance (USD99.67), added
recover_logic5_balance.py and focused tests. The stopped campaign had 36 complete
endpoints (all arms seeds1-9), twelve SIGTERM credit-outage interruptions (seeds10-12)
and 32 untouched jobs (seeds13-20). Retained endpoints are audited and hashed;
all twelve interrupted attempts restart from epoch zero in new directories, not
unsafe asynchronous checkpoint continuation. New output locations are the only
run-command change; frozen scientific sources and original artifacts are intact.

Twelve-way dispatch in tmux logic5-balance-recovery, with Comet unchanged.
Single recovery plan/status index on /local1, replacement outputs there, and
first-attempt outputs on /local0 to share storage capacity. Both filesystems are
checked before dispatch. Provider credit checks and visible failure holds remain;
no automatic retries. Original balance_recovery.json points to the new index.
The user allowed invalid-data cleanup if needed; no data was deleted at restart.
Failed-attempt evidence and costs must remain available if bulk artifacts are
pruned later. Details: reports/LOGIC5_BALANCE_RECOVERY.md.

29 initial focused tests passed, followed by six recovery tests including an
additional twelve-launch/36-retention controller regression (30 distinct checks).
No unrelated failures or ordinary invalid LLM responses qualify for replacement.

## 2026-09-24: Completed Logic-5 Paired Endpoint Statistics

Added analyze_logic5_four_arm.py; analysis only, no new simulations/model calls,
source database changes, experimental source edits or manuscript changes.
All 80 recovered/retained endpoints passed saved-artifact identity, completion,
budget, drained-stream and provider-balance checks. Terminal unions were
cross-checked with completion metrics. Results live in
reports/logic5-four-arm-statistics-20260924 (PDF/PNG, report, endpoint/test CSV,
JSON with paths and hashes).

Two-sided exact Pratt signed-rank tests on raw differences, Holm correction over
all six arm pairs and both endpoints (12 tests); 20,000 paired seed-bootstrap
samples for effect intervals. Reused existing paired summaries and Holm helper.
Slow explicit sign enumeration was replaced locally by the mathematically same
finite rank-sum distribution; regression tests agree with exhaustive SciPy
permutations for ties/zeros and standard exact Wilcoxon for 20 untied differences.
Nine analysis tests passed. Intervals are individual, not simultaneous, and the
post-hoc/symmetry assumptions and unmatched asynchronous schedules are explicit.

Full-duplex median paired loss reductions: 47.4% vs none (Holm p=.001154),
27.0% vs insertion-only (p=.000755), 24.6% vs return-only (p=.001678).
There is no corresponding increased qualifying-topology yield: full duplex has
45 fewer on average than insertion-only (p=.03289); its count differences vs
none and return-only do not survive correction. This supports rank-one quality
in this system/horizon, not universal portfolio expansion or proof of a factorial
synergy mechanism. Counts remain distinct from structural-distance diversity,
truth-table correctness and robustness. PDF/PNG layout was visually inspected.

## 2026-09-24: Logic Portfolio Pruning and Perturbations

Added run_logic5_portfolio_diagnostics.py and analyze_logic5_portfolio_diagnostics.py
as separate post-hoc analysis; original search sources, archives and manuscript
remain untouched. Read-only union extraction matched all80 completed endpoints:
18,283 qualifying topology-by-run records,15,856 unique numerical replay jobs.
Order-invariant topology identities remain distinct from order-sensitive replay
keys; the cache can retain tiny CVODE roundoff differences from earlier orders.
Every stored score is preserved. Replay deviations beyond the prior strict
tolerance are logged; a documented absolute1e-7 solver floor is used, with any
larger discrepancy or changed classification/qualification aborting visibly.

Reuses exhaustive32-subset Logic pruning at epsilon0.01 (.001/.05 sensitivity),
all16 correctness, unchanged surviving rates/template and canonical trajectories.
Applies32 shared antithetic 10% rate draws to all valid originals and primary
cores; saved winners also receive the prior five radii and10 OAT probes. No
clipping, fitting or LLM calls. Families use complete added-reaction structural
tokens and average-linkage Jaccard cut0.60, with cutoff sensitivity. Per-seed
counts, survival, clustered-core recurrence, medoids and plots are generated
automatically after all diagnostics complete, with18 corrected post-hoc count
contrasts and explicit robustness/structural-family caveats.

13 focused tests passed. Two real CVODE cases exactly reproduced baseline losses
and completed404 calls total; their persistent results are reused. Launched64
workers in tmux logic5-portfolio-diagnostics, with Comet progress and automatic
final figure upload. Frozen plan, extracted selection, per-candidate checkpoints
and status are in /local1/rossin/search-exploration/logic5-portfolio-diagnostics-20260924.
Protocol and operational details: reports/LOGIC5_PORTFOLIO_DIAGNOSTICS.md.
No invalid or valid original run artifacts were removed.

## 2026-09-24: Canonical-Order Diagnostic Revision

The saved-order Logic diagnostic study failed visibly after680 jobs: added
IDs[246,146,326,301,408] at unit rates produced a CVODE failure although its
saved loss was0.03868613969493665. Ascending ID order replayed0.03868614838872502
with all16 correct, confirmed in both evaluator paths. Following explicit user
approval, diagnostics now sort added IDs together with their rates before all
baseline, subset and perturbation calls. The fixed template is unchanged.
Canonical numerical keys deduplicate equivalent permutations, retaining all
original loss references and saved orders. No tolerance was further relaxed.

New output logic5-portfolio-diagnostics-canonical-20260924 is isolated from the
failed saved-order attempt; no previous numerical diagnostics are reused.
The new plan records the previous plan hash and failure. The original study
also retains runner_saved_order_snapshot.py. Comet experiment names now use
the output directory so protocol revisions remain distinguishable. Focused
tests check permutation invariance, rate alignment and canonical ordering in
every mocked subset/perturbation evaluation. This is post-hoc numerical
standardization, not evidence that these networks are order-robust. Original
search results and endpoint statistics remain unchanged.

Verification:13 focused tests passed; the formerly failing candidate completed
all32 subsets and64 perturbation records under canonical ordering, loss
0.03868614838872502, all16 correct. New extraction retained18,283 entries and
deduplicated to9,753 numerical jobs. Two fresh launch smoke jobs passed404 calls.
Started64 workers in tmux logic5-diagnostics-canonical with Comet experiment
ea6fdd591c4448dabd169917554ff92f; final completion is not asserted here.

## 2026-09-24: Nonblocking Recorded-Order Numerical Recovery

Canonical-only diagnostics stopped at114/9,753 candidates on a second network
that succeeds in saved order. User approved trying both without blocking the
cohort. Added deterministic canonical-first, then lexicographically ordered
recorded-order fallback, with rates kept aligned and subset order projection.
Baseline success still requires all reference and correctness checks; draw and
subset retries are numerical-failure-only, never loss/accuracy cherry-picking.
All attempts remain logged. Unresolved baselines are explicit processed records,
not invalid designs, and do not abort other candidates. Recovered/unresolved
counts are tracked in status and Comet. Genuine unexpected exceptions still
surface rather than being hidden by a blanket catch.

Analysis reports unresolved counts and observed count lower bounds, suppresses
count significance tests if any baseline is unresolved, and restricts winner
curves to matched resolved seeds with denominators stated. Empty/unresolved
reports render without crashing. New isolated output is
logic5-portfolio-diagnostics-order-fallback-20260924; previous protocol records
are not reused. Previous runner and analyzer snapshots were preserved in the
canonical-only directory. Original experiments and endpoint statistics unchanged.

18 focused tests passed. Both previously failing real networks completed their
full diagnostics (125 evaluator calls total); one needed canonical order, the
other recovered via a recorded order. Prepared18,283 entries/9,753 jobs and
launched64 workers in tmux logic5-diagnostics-fallback. Comet experiment:
4e4630d25e2342ae8becda566a4fe822. Plan SHA256:
1e28309f64565941be578982a0da36bcfb77b2690f3496fdf5e8878d4b19a3db.

## 2026-09-25: Portfolio Robustness Figure

Added plot_logic5_portfolio_robustness.py and a separate PDF/PNG with caption,
hashed source provenance, summary JSON and seed-level CSV under
reports/logic5-portfolio-robustness-20260925. Panels show total distinct cores,
robust cores, within-seed robust fractions, and all three full-duplex paired
robust-core contrasts. All four arms and20 seeds remain visible. Reused the
existing18-test Holm family, rather than re-testing a selected favorable pair;
bootstrap intervals are explicitly marginal. No numerical or model calls and
no changes to frozen analyses or manuscript.

Full versus none:16.30 versus9.50 robust cores per seed (+71.6% in ratio of
means), Holm p=.031174. Mean within-seed robust fractions:8.7% versus4.8%,
descriptively. Directional-arm contrasts are not statistically resolved. Caption
distinguishes tested parameter tolerance from proven biochemical motifs, notes
variable core dimension and observed-rate selection, and retains the separate
winner-robustness caveat. Six focused tests passed; rendered layout inspected
and adjusted to remove title/subtitle overlap.

## 2026-09-25: Explicit Canonical Topology Recount

Added plot_logic5_canonical_topology_counts.py to independently reconstruct
reaction-multiset identities from the frozen library/template and all9,753 saved
diagnostics, including pruned structures. Ignores reaction order, rates and
library-ID aliases; preserves type, labelled species, stoichiometric multiplicity
and input channels. Compared per-seed raw/core/robust-core counts against all80
previous endpoints: zero discrepancies. Existing topology hashes already removed
permutation/rate duplication. No experimental records or original statistics
changed. This is not quotienting by species renaming or functional equivalence.

Added LOGIC5_CANONICAL_TOPOLOGY_COUNTS.pdf/png alongside the portfolio-robustness
figure, plus structural audit JSON and per-seed/pooled CSVs. The two-row figure
distinguishes seed-level counts from unions across20 seeds. Pooled robust cores:
none107, inward106, outward152, full139. Thus full has the largest mean per-run
robust count, but outward has the largest pooled union; pooled counts receive
no significance test. Across all arms/seeds:4,357 valid original topologies,
4,354 cores,294 robust cores. Pooled core counts may exceed originals because
different parameterizations of the same original can prune to different cores.

Updated the portfolio-robustness figure labels/caption to state unique canonical
topologies and checked its source hash and counts against the new audit before
rendering. Nine focused tests passed, covering reaction/parameter permutations,
input modulation, multiplicity, species roles and recurrence across seeds.
Both figure layouts were visually inspected.

## 2026-09-25: Robust-Core Provenance Audit

Added analyze_logic5_robust_provenance.py and reports/logic5-robust-provenance-20260925
with PDF/PNG, summary JSON, per-core/per-witness attribution CSVs, parent checks,
per-seed and pooled counts, and interpretation. Read all80 completed databases
immutably and matched exact selected rate-vector hashes to search journal events
and direct supplied-rate LLM banks. Checked1,099 robust witnesses and reconciled
all1,064 seed/core entries to the prior canonical robustness counts. No unresolved
provenance; no new simulation, fitting or model calls. Current design rows can
overwrite earlier provenance, so route attribution uses journal/bank evidence,
with shared paths explicitly grouped rather than given arbitrary credit.

Full-duplex exclusive mean cores/seed: direct LLM1.15, LLM+AdaGrad8.70,
RL+AdaGrad6.10, RL-lineage mutation0.05, LLM-lineage mutation0.05, multiple0.25.
No exclusively unrefined RL-sampled selected witnesses. The gain is concentrated
in fitted LLM proposals descriptively, not proof of EIL causality or optimizer
necessity for robustness. Policy imitation and HOF conditioning do not produce
explicit ancestry links; parent robustness was not newly tested. Mutation may
include subsequent rate fitting; saved parent checks expose unchanged-topology
fit steps and identical-parent cases. Six focused tests passed. Figure inspected.

## 2026-09-25: Paired Directional Component Checks

Added analyze_logic5_directional_components.py and a separate report/JSON/CSV
under reports/logic5-directional-components-20260925. Audited all80 run journals
for best RL-emitted loss, separately before and allowing AdaGrad refinement;
excluded LLM/GA events and incomplete solves. Used robust provenance route sets
to count LLM-supported canonical cores once per seed, including shared routes.
No new simulations or model calls. Eleven focused tests passed, including
event-source versus mutable-design-source attribution and fit separation.

Four one-channel edges x three endpoints yield12 explicitly post-hoc paired
tests. Report both Holm12 and a sensitivity family combining the18 previous
portfolio tests (30 total), not a claim of adjustment across the entire project.
Outward feedback raises LLM-supported robust cores: return-only vs none9.65 vs
3.70 (16/0/4, rawp=.002195, Holm12=.02415, Holm30=.06147); full vs insertion-only
10.15 vs4.75 (16/2/2, rawp=.000412, Holm12=.004944, Holm30=.01236).
RL-sampled initial-rate endpoints are almost identical; no corrected difference.
RL+fit full vs return-only:15/2/3, median paired ratio.89718, rawp=.02792 but
Holm12=.27924/Holm30=.67017. Thus the outward LLM-lineage yield evidence is
stronger than evidence of improved RL-emitted loss in this cohort.

## 2026-09-25: Per-Channel Logic and RPA Figure

Added plot_task_channel_effects.py and reports/channel-effects-logic-rpa-20260925
with a two-page PDF, individual vector/PNG figures, contrasts CSV, source-hashed
JSON and scope report. Each channel is contrasted with the other channel off
and on. Logic reuses the audited robust-core provenance and Holm30 correction.
RPA direct supplied-rate qualifying LLM yield is explicitly a different metric:
whole-portfolio RPA robustness is not available. RPA uses all completed repaired
directional runs and the historical/reused independent union, so its comparisons
are descriptive and disclose scheduling/control limitations. RL endpoints use
event sources, exclude direct LLM/GA events, and distinguish with/without fitting.
Older no-GA cohorts are not pooled. No new evaluations or model calls.
Four focused tests passed (direct-yield filtering/deduplication, effect and
interval orientation, RL event attribution and missing-data rejection).
Both rendered figure pages were visually inspected.
