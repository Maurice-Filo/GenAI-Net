# Parity-12 GA Pilot

Campaign: `parity12-ga-trial-100epoch-seeds0to1-20260910`.

This isolated pilot uses seeds 0 and 1 and the first 100 epochs of the
completed Flash insertion-only source runs. It does not replace the frozen
300-epoch GA cohort or change the long RL campaigns' hyperparameters.

## Configuration

- Twelve added reactions; unchanged canonical parity evaluator and CVODE.
- 100 batches of 1023 search candidates: 102300 search evaluations per seed.
- Exactly 50 saved Flash proposals per seed, replayed at their original
  evaluation boundaries within that prefix: 102350 candidate evaluations total.
- No new model calls. Proposals arriving after the prefix are excluded.
- Two simultaneous seeds, eight CPU workers each, normal process priority.
- Mutation probabilities: rates 0.45, reaction replacement 0.45, random 0.10.
- Log-rate mutation sigma 0.5; initial rates [0.1, 50]; mutated rates positive
  and unbounded, matching the existing Logic GA baseline.
- Archive size 30. Track loss, number of designs correct on all 16 truth-table
  rows, maximum archive rate, and rates outside the LLM bounds [0.001, 100].
- Comet project: `redsnic/genai-net-ga-fixed-proposals`; experiment names start
  `parity12-ga-PILOT-100epoch-seed`.

Source audits, frozen replays and hashes, plan, status, logs and run outputs are
under `campaigns/parity12-ga-trial-100epoch-seeds0to1-20260910/` beside this file.
The worker reevaluates each paired source's 30 checkpoint designs after search
to verify losses and count correct truth tables. These validation-only
evaluations do not update the GA and are recorded separately from its budget.

## Interpretation Limits

This is a two-seed pilot, not inferential evidence from a completed cohort.
The source RL sampler permits the null reaction whereas GA and direct LLM
proposals exclude it. The comparison is therefore not fully proposal-space
matched. Existing artifacts are retained, and this discrepancy is not masked.
GA rate mutation is unbounded; it must not be described as a bounded-rate
comparison. Source prefixes retain their original RL hyperparameters.

## Operations

The pilot runs in tmux session `ga-parity12-pilot`. Attach with:

```bash
tmux attach -t ga-parity12-pilot
```

Logs are written to the campaign's `queue.log`, `seed0.log`, and `seed1.log`;
each run writes `progress.csv` and, on success, `completed.json`.
The launcher refuses automatic restarts or output overwrites.

The four long parity RL campaigns and deferred full parity GA watcher were
suspended separately. The operational record is
`comparisons/llm_seeded_evolution/analysis/control/parity-pause-20260910.json`.
Their historical status JSONs may still say running; the suspended processes
are authoritative. Logic GA and the web interfaces were not suspended.
Suspension retains RAM and GPU allocations, is not a durable checkpoint, and
does not stop remote provider deadlines. Review pending requests and timeouts
before resuming; do not launch another copy over the same outputs.
