# LLM-seeded evolutionary search

An independent baseline for the paper's Logic, RPA, and Classifier comparisons. It imports
the shared task, candidate validator, CVODE evaluator, and topology serializer,
but does not construct an RL policy, update model weights, or call an LLM.
Comet logging is optional via `--comet-project`; local artifacts are always
written. Search code, replay handling, settings, and tests live in this folder.

**Default proposal source: No communication.** By default, each seed loads its
audited local-Qwen No communication run for Logic/RPA, or hosted DeepSeek Flash
No communication run for Classifier, from the paper registries. The exporter
requires `communication_mode=none` and rejects any explicit enabled direction.
Legacy manifests omit the individual direction flags; their explicit `none`
mode supplies those flags. Without explicit opt-in, insertion-only runs are
rejected. `--source-mode llm-to-rl` admits an explicitly supplied insertion-only
source for matched-optimizer comparisons. Outward-only and full-duplex runs
remain inadmissible. The LLM's own evaluator feedback remains part of the source run.

The source is read without modification. The exported `replay.json` contains
the original candidates, losses, topology hashes, completion boundaries,
source fingerprints, scientific configuration, and original terminal archive.
It can be reused with `--replay /path/to/replay.json` without accessing the
original database. Database hashes are checked against the audited registry
when using automatic source selection.

## Algorithm and accounting

- Select parents uniformly from the best 30 labelled topologies, preserving
  only the best parameterization per topology. Capacity is inherited from the
  source run rather than independently hard-coded.
- With probability 0.45, perturb one added reaction's rate by multiplying it by
  `exp(Normal(0, 0.5))`.
- With probability 0.45, replace one added reaction with an allowed absent
  reaction and initialize its rate log-uniformly on `[0.1, 50]`.
- With probability 0.10, draw an entirely new admissible topology and rates.
  Random search also bootstraps an empty archive.
- Fixed template reactions and the null reaction are excluded from mutations
  and random sampling. Every candidate uses the source's exact reaction budget.
- Search-generated rates have positive support; they are not clipped to the
  direct LLM interval. Saved LLM proposals retain their original clamped rates.
- Produce batches of 32 offspring from the current archive, then evaluate and
  update it. The batch size is a scientific setting independent of worker count.
  Results are consumed in proposal order for deterministic parallel execution.
  The isolated fixed-insertion campaign below uses 1,023 instead of this default.

Replay availability follows the **completed epoch**, not the request epoch.
In the original trainer, a response is merged after the epoch's batch has been
evaluated. Thus a response completed in epoch 7 enters after `8 * 1023 = 8184`
search evaluations. Evolutionary batches are split at these boundaries. Future
proposals are never used early. Missing/rejected members receive no replacements.

A full run has 102,300 evolutionary evaluations plus the original run's actual
accepted LLM evaluations (at most 50), matching the source's counted candidate
budget. Replayed proposals are reevaluated and their losses and hashes must
match the saved values. Unexpected validator/infrastructure failures abort;
canonical simulator failures keep the evaluator's standard finite failure loss.
The optional validation-only command is a separate reproducibility check, with
its own evaluation count; it is not part of an optimization run's budget.

## Run the Logic example

From the repository root, first check all proposals from seed 0:

```bash
.venv/bin/python -m comparisons.llm_seeded_evolution.run \
  --task logic --seed 0 --validate-only --workers 2
```

A pilot preserving the original schedule through the first proposal arrival:

```bash
.venv/bin/python -m comparisons.llm_seeded_evolution.run \
  --task logic --seed 0 --search-budget 10230 --workers 4
```

In this No communication seed, the first ten proposals become available after
8,184 search evaluations. The pilot therefore includes 2,046 search evaluations
after their insertion. It is explicitly marked `pilot=true` and cannot enter
the full-budget paper comparison.

For one full-budget seed, omit `--search-budget`:

```bash
.venv/bin/python -m comparisons.llm_seeded_evolution.run \
  --task logic --seed 0 --workers 4
```

The algorithm is also selectable in the existing comparison runner:

```bash
.venv/bin/python comparisons/rpa_search/scripts/run_genai_julia_20seed_batch.py \
  --task logic --method llm_seeded_evolution \
  --seed-start 0 --seed-end 19 --n-cpus 4
```

Runs execute sequentially, with the requested workers per run. Full completed
baseline runs are recognized using their actual replay budget rather than the
legacy 102,400-RL-only cutoff. An interrupted/nonempty output directory is never
silently overwritten or treated as complete; use a new `--run-id` to restart.

Outputs go under `comparisons/rpa_search/data/raw/llm_seeded_evolution/<run_id>`:

- `replay.json`, `config.json`, `run_manifest.json`: reproducible inputs.
- `candidates.csv`: loss, rates, topology hash, operation, parent, and proposal lineage.
- `progress.csv`: search and LLM evaluation counts, best loss, and qualified yield.
- `terminal_archive.json`, `best_network.json`, `best_network.txt`: final designs.
- `completed.json`: completion, resource counts, operation counts, and pilot status.
- `validation.json`: only for validation-only runs.

Run IDs are separate from the existing experiments. No paper claims or frozen
confirmatory analyses are updated by running this baseline.

## Paper comparisons and plots

Classifier uses the same evolutionary settings, with the archived task's six
added reactions, 32 initial-condition scenarios, and CVODE tolerances of `1e-6`.
Candidate counts are the comparison axis; legacy breadth CSVs incorrectly label
the inert input slot as one scenario, whereas this runner counts all 32.

```bash
.venv/bin/python -m comparisons.llm_seeded_evolution.run \
  --task classifier --seed 0 --validate-only --workers 4
.venv/bin/python -m comparisons.llm_seeded_evolution.run \
  --task classifier --seed 0 --search-budget 10230 --workers 4
```

The first Classifier seed-0 batch arrives after 5,115 search evaluations, leaving
5,115 further evolutionary evaluations in this pilot (10,240 total including
the ten proposals). Later batches arrive after 25,575, 45,012, 67,518, and 86,955
search evaluations. Proposal lineage distinguishes direct LLM contributions
from their evolved descendants and from the independently discovered designs.
Classifier is also selectable with `--method llm_seeded_evolution` in the batch
runner. Other methods in that legacy runner remain restricted to Logic/RPA.

After a pilot completes, generate a matched-candidate-budget comparison with
the frozen RL-only control and a plot of the direct proposal contribution:

```bash
.venv/bin/python -m comparisons.llm_seeded_evolution.pilot_report \
  comparisons/rpa_search/data/raw/llm_seeded_evolution/classifier_search10230_seed0_flash_none_replay_v1
```

This writes `pilot_comparison.json`, `.png`, and `.pdf` in the run folder. The
report checks the RL trajectory against its frozen audit hash and distinguishes
the winner's proposal ancestry. It does not promote pilots into full-budget tables.

For full-budget Classifier results, the paper exporter accepts
`--task classifier --methods no_communication llm_seeded_evolution` (also its
Classifier default). The local-model communication-arm tables cover Logic/RPA
only, so they are not mixed into the Flash Classifier comparison.

After a full seed completes, export an optional table:

```bash
.venv/bin/python paper/iclr2027_genai_net_llm/compare_search_algorithms.py \
  --task logic --seeds 0 \
  --methods no_communication llm_to_rl llm_seeded_evolution
```

Omit `--seeds` for all 20 seeds. Choices are `no_communication`, `llm_to_rl`,
`rl_to_llm`, `full`, and `llm_seeded_evolution`. The script writes CSV, JSON,
and a LaTeX table under `paper/iclr2027_genai_net_llm/generated/llm_seeded_evolution`.
It rejects pilots, missing seeds, mixed search protocols, and changed inputs.

In this default export, evolution and No communication use the same model proposal stream. Their
comparison asks whether evolutionary refinement beats the independent RL+LLM
terminal union. Published communication arms are useful contextual comparisons
but used **different recorded proposal streams**. The exporter labels this
difference explicitly and computes matched fixed-stream differences only
against No communication. The separate fixed-insertion campaign below instead
replays insertion-only proposals into evolution, allowing comparison directly
with the recorded source RL run without rerunning RL.

The existing generic plotter recognizes `llm_seeded_evolution`. To plot a run,
pass its generated `config.json`:

```bash
.venv/bin/python comparisons/rpa_search/scripts/plot_results.py \
  --config comparisons/rpa_search/data/raw/llm_seeded_evolution/logic_search102300_seed0_local_none_replay_v1/config.json \
  --paper
```

## Fixed-insertion-stream Logic campaign

The 2026-09-09 comparison uses **20 local-Qwen and 20 Flash Logic seeds**,
replaying each corresponding **insertion-only RL run's** candidates at its
recorded completion-epoch boundaries. GA and its RL comparator therefore share
the proposal bank, rates, and evaluation-count availability schedule, unlike
comparisons between independently collected communication arms.

The algorithm is mutation-only elitist evolutionary search, not a crossover GA.
It keeps the default 45% rate mutation, 45% reaction replacement, 10% random
restart, top-30 archive, and initial log-uniform rate prior. The offspring batch
is **1,023**, with **100 batches / 102,300 search evaluations** plus each source's
actual accepted LLM evaluations. Offspring use the archive at batch start;
replay enters after the recorded completed epoch's batch. Rates and canonical
CVODE settings are unchanged. Initial populations and initialization priors are
optimizer-specific; same seeds do not imply identical random populations.

The active campaign folder is:

`campaigns/logic-insertion-fixed-b1023-20260909-local/`

Its `plan.json` records the four-run concurrency, four CPU workers per run,
**normal priority (no nice adjustment)**, source/context hashes, and separate
Comet project `genai-net-ga-fixed-proposals`. No GPUs or new LLM calls are used.
The directory suffix does not restrict the model cohort: both sources are included.

Every source database matches its audited hash, and all 200 saved request HOFs
are empty as required for insertion-only communication. All 40 seeds preserve
the initial-HOF withholding. Failed rounds in local seeds 1, 4, and 15 contribute
no replacement proposals; null returned/accepted fields are preserved, and
total replay evaluations reconcile with the source terminal counter. Each
candidate is reevaluated at replay time and must match its recorded loss/hash.

Local seed-0 validation reproduced all 48 proposals, and Flash seed-0 reproduced
all 50, with zero absolute loss difference. These validation-only evaluations
are separate from the optimization budget. Tests also check 1,023-offspring
batch semantics, source isolation, failed rounds, and strict comparison inputs.

Monitor `status.json`, per-run `progress.csv` and logs, or the
[Comet project](https://www.comet.com/redsnic/genai-net-ga-fixed-proposals).
The tmux session is `ga-fixed-logic-1023`. The queue stops dispatching new jobs
after a scientific or infrastructure failure; incomplete outputs are not
automatically reused. Existing parity campaigns are not controlled by this queue.

```bash
# Observe the scheduler log without changing the running experiment.
tail -f comparisons/llm_seeded_evolution/campaigns/logic-insertion-fixed-b1023-20260909-local/queue.launch2.log

# Refresh the comparison using completed, hash-verified seeds only.
.venv/bin/python -m comparisons.llm_seeded_evolution.fixed_stream_report \
  --root comparisons/llm_seeded_evolution/campaigns/logic-insertion-fixed-b1023-20260909-local
```

The comparison refreshes after each completion. It compares GA with the actual
source RL archive, not another arm's proposal bank, and produces `endpoints.csv`
and `comparison.json`. Inferential testing waits for all 40 seeds: two metrics
(loss and qualified yield) across two model sources, one exploratory four-test
Holm family. Means use paired-seed bootstrap intervals. Budget-resolved losses
and qualified counts are retained in each `progress.csv` for trajectory analysis.
This is a conditional optimizer comparison, not a selective-replay ablation or
a new interactive LLM feedback experiment. It does not alter manuscript results.

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q comparisons/llm_seeded_evolution/tests
```

Tests cover replay chronology and separate budgets, missing members, archive
deduplication, parameter improvement, mutation validity, deterministic batching,
strict rejection of communication branches, source loss parity, and runner selection.

For the optional real-CVODE functional smoke test, set `GENAI_EVOLUTION_REPLAY`
to an exported No communication `replay.json`. That test deliberately supplies
the first batch at smoke-test step zero and performs 64 evolutionary evaluations.
Its temporary outputs are not paper evidence. This does not change the production
runner's original completion schedule.
