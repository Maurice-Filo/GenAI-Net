# RPA Flash + AdaGrad Campaign

Started 2026-09-17 after approval and live Harness verification.

## Active Cohort

- Root: `/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1`
- tmux: `rpa-flash-adagrad-20260917`
- Seeds 1-20; eight concurrent, twelve initially queued.
- 100 epochs; 1023 search proposals/epoch; NN intermediate size 1024.
- 214793 total canonical/joint-evaluation cap per seed. Search reserves 102300;
  LLM final evaluations and AdaGrad share the remainder, with exact caching.
- Six single-thread search workers and two persistent Julia fit workers per
  seed: 64 numerical slots across eight seeds. Two additional canonical worker
  processes per seed verify fits and evaluate LLM candidates; they normally
  replace a Julia solve, but LLM evaluation creates transient extra demand.
- Model: `deepseek-official/deepseek-v4-flash`; unchanged workspace-loss system
  and Decider/Writer role instructions, approved versioned RPA task prompt.
- Full communication; initial HOF/EIL/exclusion and exploration context withheld.
- Ten candidates per scheduled round at epochs 0,5,...,95; one pending round per
  seed. Busy cadence points are skipped and logged. Each round has Decider and
  Writer invocations; native-tool turns/continuations are additional API calls.
- No scientific wall-clock cutoffs; existing 900-second Harness role deadline.
- No RAG, exploratory agent evaluations, sparsity penalty or post-pruning.
- Comet: https://www.comet.com/redsnic/genai-net-exploration
  Each run saves its experiment ID in `comet.json`.

## Data Flow

Background Harness generation and canonical evaluation do not block search
epochs. Valid completed candidates merge into both the HOF and discovery
archive at epoch boundaries. They compete with current search proposals for
AdaGrad admission under the existing rule, not a separate preferential quota.
Structurally valid proposals with incomplete initial trajectories are not
inserted as successful designs, but remain eligible for fitting/rescue.

Fitted candidates retain their source/origin, parent identity and explicit
`julia_adagrad` operator. Current HOF states, including fitted structures/rates,
are exposed on subsequent requests. Completed LLM outcomes also update the
agent's persistent self-memory. The policy receives immediate search rewards;
delayed fitting and LLM insertion affect future sampling and imitation.

Terminal draining records separate training and terminal boundaries. Late
arrivals cannot be credited with improving already completed RL updates. The
terminal DB counters include LLM evaluations. Failures remain in logs/history;
numerical/infrastructure errors hold further seed dispatch, not silent retries.

## Checks and Initial Observation

- Live readiness: 10/10 valid proposals, zero initial HOF entries, two role
  invocations, zero exploratory evaluations, 468.94 seconds. Best canonical
  supplied-rate loss: 0.005954755380383372. This is a readiness batch, not a seed
  in the campaign or a statistical performance claim.
- All eight first campaign workspaces were audited: correct Flash model,
  workspace-loss protocol, zero HOF entries, no exploration-state file and
  explicitly withheld EIL status.
- All eight reached epoch 1 while their first LLM round was still pending,
  demonstrating concurrent search in the live path.
- Initial two-second utilization sample: 48.3 CPU-core equivalents during
  startup/calibration, with 184.5 GiB available RAM. This is a snapshot, not a
  claim that all 64 slots are continuously busy.
- Tests: 34 passed / six optional Julia tests skipped in the initial combined
  suite; three selected real Julia budget/continuation tests then passed. The
  final new integration/readiness suite passed all seven tests. Another 35-test
  run covered the integration plus the existing Harness client/protocol.
- Frozen campaign source/input hashes and plan digest matched after launch.

## Startup Failure Preserved

The first directory, `rpa-flash-adagrad-20seed-20260917`, contains eight failed
startup attempts. Comet rejected an inherited environment credential before any
search or campaign LLM call. Credential echoes in those logs were redacted;
all failure diagnostics remain. The existing local Comet configuration then
passed an authentication-only check. The new controller removes the stale
`COMET_API_KEY` environment override and uses that configuration. The fresh
`-r1` cohort keeps the same scientific settings, with new output paths; no
failed attempt is presented as a completed experimental replicate.

## Comparison Caveat

The reference is the completed 20-seed no-LLM AdaGrad workflow. Seeds,
scientific hyperparameters and total budget cap match, but per-seed workers
changed from 32 search / 8 fitting to 6 / 2. Timing affects asynchronous fitting
and LLM insertion opportunities. Report actual evaluations, fits, request counts
and insertion epochs; do not claim identical proposal schedules or equal CPU
cost merely from the common evaluation cap.

## Monitoring

`progress.json` contains all seed states and current metrics; per-seed
`status.json`, `results.sqlite`, `exploration.sqlite`, `llm_history.json`,
`harness-workspaces/` and `julia_jobs/` retain detailed artifacts.

```bash
tmux attach -t rpa-flash-adagrad-20260917
```

The plan contains exact commands, prompt/source hashes and configuration files.
Creating `HOLD` in the active root pauses new dispatch while active seeds finish.
No automatic controller resume or failed-seed retry is enabled.
