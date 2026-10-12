# Validation on September 8, 2026

The proposal source is the local-Qwen **No communication** Logic seed 0 run.
Its explicit mode is `none`; no RL-derived context or insertion-enabled branch
is admitted by the replay loader.

**Recorded-proposal parity:** all 50 proposals were reevaluated with the shared
CVODE evaluator. All topology hashes matched, and the maximum absolute loss
difference was **0.0**. No new model calls were made. The validation artifact is
`../rpa_search/data/raw/llm_seeded_evolution/logic_seed0_none_replay_check_v1_validation/validation.json`.

**Functional smoke test:** 18 tests passed, including a real CVODE run using
the first ten saved proposals and 64 evolutionary evaluations. The operation
counts were 31 rate mutations, 25 reaction replacements, and 8 random restarts.
The best loss decreased from **0.3720332693** to **0.1652457213**. This smoke test
deliberately exposes the first batch at test step zero; it verifies the software
and is not a paper-budget experiment or evidence of comparative superiority.

**Regression checks:** the subsequent lightweight run passed 22 tests and
skipped the optional real-CVODE test, which had already passed separately.
Python compilation and whitespace checks passed for the changed entry points.

**Original-schedule pilot:** started as
`logic_seed0_none_evolution_pilot_v1` with 10,230 search evaluations and four
workers. Its first ten model proposals are scheduled after 8,184 evaluations,
leaving 2,046 evaluations for refinement. The pilot is explicitly excluded from
full-budget paper tables. Its live progress and eventual completion are at:

- `../rpa_search/data/raw/llm_seeded_evolution/logic_seed0_none_evolution_pilot_v1/progress.csv`
- `../rpa_search/data/raw/llm_seeded_evolution/logic_seed0_none_evolution_pilot_v1/completed.json`

The pilot subsequently completed in **2,917.44 seconds (48.6 minutes)** with
10,230 search evaluations plus 10 replayed proposals. Final best loss was
**0.0355527184**, with **3 qualified topologies**. None of the first ten LLM
proposals entered the archive: their best loss was 0.3720, while the worst of
the top 30 before their arrival was 0.06892. These improvements came from
evolutionary search in this seed.

The winning design's rates were approximately `[13.3251, 10.4470, 0.99284,
10.5829, 3.77154]`. Reevaluation reproduced its loss exactly. Tightening both
CVODE tolerances from `1e-8` to `1e-10` returned **0.0355526537**, an absolute
change of approximately `6.47e-8`. These two post-run validation evaluations
are separate from the optimization budget and recorded in `parameter_sanity.json`.

This is a single-seed, one-batch pilot. No full-budget performance conclusion
is recorded here.

## Classifier extension

The Classifier seed-0 source is the audited **DeepSeek Flash No communication**
run. All 50 saved proposals reproduce with identical topology hashes and
**zero absolute loss difference** (10.80 seconds, four workers). This is saved
in `../rpa_search/data/raw/llm_seeded_evolution/classifier_seed0_none_evolution_parity_v1_validation/validation.json`.

The canonical task uses six added reactions, two species, 32 initial conditions,
and CVODE `rtol=atol=1e-6`. The inert input slot is not counted as a scenario.
The extended regression suite passed **25 tests**, with the optional CVODE test
skipped in that invocation. The real Classifier smoke test separately passed:
64 evolutionary evaluations plus ten proposals, comprising 32 rate mutations,
24 reaction replacements, and eight random restarts. Its best loss remained
the starting proposal loss, **0.0076663225**. As with the Logic smoke, proposal
availability was moved to step zero only for this software test.

The real pilot `classifier_seed0_none_evolution_pilot_v1` retains the source
schedule: its first ten proposals arrive after 5,115 search evaluations, and
its planned total is 10,230 search plus ten replay evaluations. All GA settings
are unchanged from Logic. The frozen RL-only seed-0 trajectory hash was checked;
its loss at 10,240 evaluations is **0.3174505597**, and its frozen full-budget
endpoint is **0.0998884884**. The first direct proposal batch alone reaches
**0.0076663225**, so a hybrid victory must not be attributed to GA without checking
the pre-proposal trajectory and subsequent refinement. The run's completion
and comparison artifacts are the authoritative result once available.
