# RPA AdaGrad Directional Communication Campaign

Launched 2026-09-17. Existing scientific sources and completed results are
unchanged. This campaign adds two communication interventions to the established
asynchronous AdaGrad/RL/GA search recipe, not a different optimizer.

## Arms and Reused Control

- **LLM-to-RL:** direct evaluated proposals enter search and may receive ordinary
  fitting, mutation and elite imitation. No RL/HOF, EIL or exclusion state is
  exposed to the LLM, including on later requests.
- **RL-to-LLM:** later requests receive evolving search context. LLM proposals
  are evaluated and retained separately; they cannot enter the search archive,
  fitting queue, HOF, visitation records or replay. Terminal aggregation combines
  the two sources without retroactively changing search.
- **Independent union:** combine each completed historical no-LLM AdaGrad/RL
  seed with that seed's direct supplied-rate LLM proposals from insertion-only.
  Do not use fitted descendants or HOF-conditioned proposals. This requires no
  additional model calls or rerun of the historical search.

The LLM retains outcomes of its own proposals in both directional arms.
Request-zero HOF, EIL, exclusions and aggregate search diagnostics are withheld.
Static task constraints and reaction masks remain available.

The independent union is a post-hoc reference, **not an exactly matched
asynchronous no-communication experiment**. Request counts, completion schedules,
concurrency and actual numerical expenditure can differ. Reusing a proposal bank
also makes donor-bank comparisons dependent. Inference costs remain attributable
to the baseline even though no extra calls are made. This is not fixed-proposal
causal replay, nor does a coupling advantage identify policy imitation separately
from mutation, archive selection and parameter optimization.

## Frozen Configuration

Each arm has seeds 1-20, 100 epochs, batch size 1023, NN width 1024,
and the same total numerical cap of 214793 as the existing cohort. Each seed
uses six search workers and two persistent Julia fitting workers, plus canonical
verification workers. Ten seed jobs run concurrently, counterbalancing arm order
within seeds. The shared hosted-Harness gate remains eight; pending model calls
do not block RL. No numerical wall-clock timeout is introduced.

Model: `deepseek-official/deepseek-v4-flash`. The approved RPA prompt and
Decider/Writer protocol are unchanged. Request cadence is every five epochs,
ten candidates per request, with at most one pending round per seed; busy
cadences are not additional queued requests. Rates are in [0.001, 100], with
exactly five added reactions and masks [0, 1, 28]. Existing validation,
continuations, role timeout and canonical CVODE scoring are retained.

## Artifacts and Monitoring

Campaign root:
`/local1/rossin/search-exploration/rpa-flash-adagrad-directional-20seed-20260917`

Tmux session: `rpa-directional-adagrad-20260917`.

Comet: <https://www.comet.com/redsnic/genai-net-exploration>.

- `plan.json`: immutable jobs, commands, configuration and source hashes.
- `progress.json`: active, queued, completed and failed jobs with latest metrics.
- `runs/*/direct_llm_proposals.jsonl`: original per-member canonical evaluations.
- `runs/*/terminal_union.json`: separate-source terminal union endpoint.
- `analysis/REPORT.md` and `summary.json`: refreshed after completions.
- `analysis/independent_union.csv`: audited reused controls as pairs complete.

Historical controls are the final `julia` attempts in
`/local1/rossin/search-exploration/rpa-statistical-20seed-20260915/active_plan.json`.
The full-duplex reference is
`/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1`
(20/20 complete at launch verification).

Union analysis verifies seed/task/bounds/budget identity, audits saved insertion-only
contexts, and deduplicates complete topology hashes. It charges historical
numerical work plus every recorded direct proposal evaluation conservatively;
an over-cap union is flagged, never silently reduced. Failed requests remain
recorded. Analysis errors are explicit. Scientific run failures stop new dispatch
while active jobs drain; there are no automatic retries. A root-level `HOLD`
file also prevents new dispatch without interrupting active jobs.

To regenerate the read-only endpoint report, from the repository:

```bash
.venv/bin/python -m comparisons.search_exploration.analyze_rpa_directional_streams \
  /local1/rossin/search-exploration/rpa-flash-adagrad-directional-20seed-20260917
```

## Verification

43 focused adapter, isolation, campaign, union and Harness/protocol tests passed.
All ten first live workspaces had empty HOFs, absent aggregate search diagnostics,
and withheld EIL/exclusion state. Own-proposal memory formatting exposes only
the LLM's proposal structures, supplied rates, losses and validity feedback.
Both old and new campaign scientific source hashes match their frozen plans.
At the startup check ten jobs were active, zero failed, with Comet logging live.
