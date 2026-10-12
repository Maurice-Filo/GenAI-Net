# Controlled no-LLM category/reward pilot

This opt-in runner compares the unmodified Julia hybrid with a combined ablation.
No existing search sources, frozen campaign sources, results, or manuscript are edited.

The ablation uses one category, reward equal to negative canonical loss, uniform
live-archive parents, loss-only topology minima and capacity eviction, and uniform
sampling with replacement from the globally lowest-loss 30 live entries. The draw
count remains min(16, number of eligible entries). StructuralLearner.update is
unchanged: canonical reaction sorting, unweighted imitation, policy-gradient
baseline, coefficients, Adam, learning rate and clipping are all inherited.

The four-dimensional context is inherited unchanged. Its coverage component is
0 before any completed evaluation, then 1/2000; the quality and budget components
retain the original formulas. Thresholds mark reporting qualification and remain
in that unchanged quality-context component. They never rank the ablated archive
or enter its reward. Config JSON remains identical within each pair; the manifest
records effective descriptor, reward and archive changes separately.

Both tasks actually use weighted tracking loss, including Logic; the generic BCE
helper is not the task used here. Canonical failure handling fills failed output
samples with 1e4 and computes the ordinary task loss. Finite incomplete outcomes
receive R=-L and never enter the live archive. There is no universal scalar failure
loss, reward clipping, rescaling or failure filtering. Missing/nonfinite canonical
losses, transport failures and evaluator exceptions stop the run without an update
or substituted reward; existing journals and failed.json preserve evidence.
Negative rewards need not produce negative centered advantages under the unchanged
moving baseline. This is explicitly counted in the analysis.

Historical RPA seed 2 has 11,135 unique incomplete canonical outcomes with losses
7.8081 to 10486.4525 (median 9917.2766). This establishes the potential reward scale
before launch, not the new pilot's outcome. The baseline regional reward has an
upper bound of 1.35 with these coefficients and no positive incomplete reward.

Historical controls are not reused. Logic's none-channel runs still made isolated
LLM calls. Older RPA code/completion protocols and absent learning diagnostics also
prevent a clean reuse. The exact task/config/worker settings come from the six
manuscript none-arm manifests. Both new RPA seed-1 arms use the current no-timeout
completion protocol already used for manuscript seeds 2/3. Both arms otherwise use
the current source, including current four-start fitting, 100 epochs, 1023 proposals
per epoch and 214793 maximum canonical-plus-joint evaluations. Following the user-authorized amendment, both tasks and both conditions use
record-only (`warn`) loss-disagreement handling. Both numerical losses, discrepancy,
thresholds and fitting diagnostics are retained; canonical loss remains authoritative.
Solver tolerances, fitting budgets and loss definitions are unchanged. Existing
Logic runs already used this setting and continue without interruption. RPA is
restarted for all three seeds and both conditions in a new campaign directory;
the previous aborted attempt and completed original baseline remain preserved.

The queue bounds concurrency to six Logic runs and one RPA run (at most 108
search/canonical/Julia worker processes, each limited to one numerical thread,
plus seven coordinators) on the observed 128-logical-CPU / 250-GiB host. Workers
per run remain exactly 6+2+2 for Logic and 32+8+8 for RPA. All logs and artifacts
are campaign-local. Failures are never automatically retried. The original
asynchronous measured-compute fitting controller remains active, so realized
fitting counts and trajectories can vary with timing. No wall-clock causal claim
is made.

Validation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SYNTH_EVO_PROJECT=/local0/rossin/git/SynthEvo/SynthEvo PYTHONPATH=. \
.venv/bin/python -m pytest -q tests/test_category_pilot.py tests/test_search_streams.py \
  tests/test_julia_search_stream.py tests/test_search_no_deadlines.py
```

Prepare a new campaign (refuses an existing directory), then execute the frozen
queue using the absolute campaign path. Every individual command is in campaign.json.

```bash
.venv/bin/python -m comparisons.search_exploration.category_pilot.prepare --output CAMPAIGN
.venv/bin/python CAMPAIGN/source/comparisons/search_exploration/category_pilot/campaign_queue.py CAMPAIGN
.venv/bin/python CAMPAIGN/source/comparisons/search_exploration/category_pilot/analyze.py CAMPAIGN
```

Analysis uses all complete committed numerical proposals and retained,
canonically reevaluated fitting results in both conditions. Internal Julia trials
are numerical expenditure, not candidate-pool entries. Supplied rates and fitted
performance are separate columns/curves; qualification counts deduplicate topology
hashes. Curves include the terminal drain. learning.jsonl retains rewards,
advantages, entropy, raw PG and imitation objectives, weighted objective scales,
pre-clipping norms, sampled elites and every supplied-rate proposal outcome.

Only paired endpoints that finish are compared. Three seeds support descriptive
pilot observations. This combined intervention cannot identify the independent
causal contribution of speciation. No pruning or perturbation analysis is run.


To adopt the already-running compatible Logic jobs and replace the RPA cohort:

```bash
.venv/bin/python -m comparisons.search_exploration.category_pilot.prepare \
  --output NEW_CAMPAIGN --reuse-logic-from PRIOR_CAMPAIGN \
  --wait-for-rpa-from PRIOR_CAMPAIGN
.venv/bin/python NEW_CAMPAIGN/source/comparisons/search_exploration/category_pilot/campaign_queue.py NEW_CAMPAIGN
```

The amended campaign links the six existing Logic run directories and records
manifest hashes and numerical/training-source compatibility. Its RPA lane waits
for any previously active RPA process to exit before launching full replacements.
Superseded pending jobs in the old queue have additive cancellation markers; no
prior frozen configurations, source archives, checkpoints or numerical evidence
are changed. Analysis uses the new campaign's selected 12 runs.
