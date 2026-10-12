# RPA Directional Campaign: Balance Recovery

## Incident and Scope

The original 40 search runs completed, but 199 of 435 request records failed
with `dsh: QUOTA: Insufficient Balance`. This was a provider-availability failure,
not an empty/invalid scientific proposal. Original aggregate endpoint reports
must not be treated as clean 20-seed communication comparisons.

The 2026-09-18 recovery restarts all affected arm-seed records from epoch zero:

- Insertion-only: seeds 12, 16, 17, 18, 19, 20.
- Return-only: seeds 13, 15, 16, 17, 18, 19, 20.

The other 27 records are reused. Original files, failures, request logs and
endpoint values remain preserved; there is no best-attempt selection and no
replacement seed. Ordinary model-output failures do not trigger replacement.
Discarded original attempts consumed 1,775,346 numerical calls; this is additional
to replacement budgets, as is any already consumed model inference.

## Execution

Recovery root:
`/local1/rossin/search-exploration/rpa-flash-adagrad-directional-balance-recovery-20260918`

Tmux: `rpa-directional-balance-recovery-20260918`.

The controller changes only each affected run's output path. Original commands,
configs, seeds, prompts, scientific source hashes, bounds, numerical budgets,
Harness roles, model and Comet project are retained. Each output name gains a
`-balance-recovery` suffix so replacement Comet experiments are identifiable.
There are up to ten concurrent runs, each with six search/two Julia fitting
workers, plus canonical verifier workers. Request-zero withholding and both
directional boundaries are unchanged.

The existing protected credential is used without printing or copying it. The
provider balance endpoint reported USD 49.56 available before launch. The new
controller checks availability before launch and every minute, and inspects
request histories and saved stderr on each polling cycle. An insufficient-balance
response holds dispatch and terminates owned active recovery process groups,
preserving their artifacts rather than allowing another silent incomplete arm.
Transient balance-check errors hold new dispatch until a successful check.
There are no automatic retries. A root-level `HOLD` file stops new dispatch
without interrupting active jobs. Scientific failures also hold new dispatch.

Monitoring: `progress.json`, `balance.json`, per-run `status.json`, and
<https://www.comet.com/redsnic/genai-net-exploration>.

## Preliminary Results

The common unaffected subset is seeds 1-11 and 14, twelve matched seeds. The
same subset is used for all rows below; no values from balance-affected attempts
are included.

| Workflow | Mean qualifying topologies | Median best loss |
|---|---:|---:|
| AdaGrad/RL without LLM | 151.17 | 0.00182550 |
| Full-duplex Flash | 173.83 | 0.00057114 |
| Insertion-only | 170.17 | 0.00058306 |
| Return-only | 177.00 | 0.00061020 |
| Independent union | 185.00 | 0.00058306 |

Full duplex versus insertion-only: topology-count W/T/L 5/1/6; loss 6/0/6.
Full duplex versus return-only: topology-count 5/0/7; loss 5/0/7.
Full duplex versus independent union: topology-count 5/0/7; loss 7/0/5.
Full duplex versus no-LLM: topology-count 9/0/3; loss 12/0/0.

The LLM-enabled workflows improve on the no-LLM reference descriptively, but
these data do not show a clear full-duplex advantage. They do not establish
equivalence or that communication is universally redundant. The independent
union reuses direct insertion-only proposals with historical no-LLM search;
it has unmatched request schedules and dependent proposal banks, not a fresh
no-communication intervention. Both directional endpoints include terminal
aggregation. No interim significance testing or new causal interpretation is
introduced here.

## Reproducibility

`plan.json` identifies every retained/replacement path and original-request audit,
with frozen inputs and controller hashes. Original `balance_recovery.json`
points to the recovery and warns about the contaminated original aggregate.
`analysis/REPORT.md` and `analysis/summary.json` rebuild matched-subset summaries
after each completion, excluding any renewed balance-affected endpoints.

```bash
.venv/bin/python -m comparisons.search_exploration.recover_rpa_directional_balance \
  --root /local1/rossin/search-exploration/rpa-flash-adagrad-directional-balance-recovery-20260918 \
  --analyze
```

Do not rerun `--prepare` or `--run` on this started root. They intentionally
reject overwrite/automatic resume. The frozen experiment implementation and
manuscript were not changed.
