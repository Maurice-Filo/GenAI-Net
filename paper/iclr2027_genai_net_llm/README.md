# High-quality CRN portfolio manuscript

Anonymous ICLR-format manuscript for the simulator-grounded RL/LLM CRN search
study. Throughout the paper, the default full-duplex model is the frozen policy
that withholds the random initial HOF, then exchanges evaluated HOF and
self-imitation-learning state from request 1 onward. SIL is reserved for
self-imitation learning; simulator-in-the-loop is written in full.

## Build

> **Results invalidated and quarantined (1 September 2026):** historical Harness
> contracts exposed fixed-template reaction IDs that native RL masked. Every
> pre-fix LLM campaign on a task with fixed template reactions is protocol-invalid,
> whether or not a stored response selected a colliding ID. The runs and all
> dependent numerical assets are isolated under the external quarantine recorded
> in `PRE_MASK_QUARANTINE.md`. No current PDF is submission evidence. The complete
> replacement protocol is in `EXPERIMENT_RECOVERY_PLAN.md`.

```bash
make
```

The default build first runs the strict artifact audit and currently stops because
the quarantined primary artifacts are absent. Numerical figures, tables, and the
compiled development PDF were intentionally removed from the active paper tree.
`make draft` also remains unavailable until replacement assets are generated;
historical prose in `sections/` is retained only to support the later rewrite.

## Regenerate analyses after contract-v2 completion

Do not run these generators against quarantined artifacts. They are retained as
analysis entry points and must first be updated to require the frozen contract-v2
fingerprint and completed replacement cohorts.

```bash
MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/python generate_paper_figures.py
MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/python generate_best_loss_trajectories.py
MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/python generate_primary_results.py
MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/python generate_quality_portfolio.py
PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/python generate_communication_mechanism_analysis.py
PYTHONPATH=../.. ../../.venv/bin/python audit_contract_v2_readiness.py
PYTHONPATH=../.. ../../.venv/bin/python ../prompt_catalog/generate_prompt_catalog.py
```

`generate_primary_results.py` requires all 160 matched primary endpoints by
default. Use `--allow-incomplete` only for monitoring; the JSON output lists
every absent artifact.

Run the repository tests with the checkout first on the import path; otherwise
Python may collect a stale user-site `RL4CRN` installation:

```bash
PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/pytest -q ../../tests
```

## Structure

- `main.tex` and `sections/`: anonymous paper and supplementary material.
- `generate_primary_results.py`: eight-task paired endpoint analysis.
- `generate_quality_portfolio.py`: matched quality-conditioned topology-yield analysis.
- `generate_communication_mechanism_analysis.py`: fixed-template-excluded structural distance, source-filtered RL-emitter analysis, and leakage checks.
- `audit_contract_v2_readiness.py`: active fail-closed manuscript build gate.
- `audit_paper_experiments.py`: forensic pre-v2 artifact and method audit; it is
  not the active build gate.
- `AUDIT_REPORT.md`: discrepancies, corrections, and residual risks.
- `PRE_MASK_QUARANTINE.md`: evidence classification and reversible move record.
- `EXPERIMENT_RECOVERY_PLAN.md`: protocol-v2 gates, campaigns, analysis, and rewrite plan.
- `generate_paper_figures.py`: architecture, initialization, and selection figures.
- `generate_best_loss_trajectories.py`: all-task default full-duplex versus RL curves.
- `generated/`: contract-v2 prompt review, static preflight, quarantine inventory,
  and future machine-generated result assets.
- `figures/`: manuscript-ready PDF figures.
- `references.bib`: primary literature used by the manuscript.
- `LITERATURE_NOTES.md` and `NOVELTY_AUDIT.md`: positioning notes.

The pre-fix prompt catalog is retained for forensic reproducibility but is not
included as active empirical evidence. Contract-v2 system, role-wrapper, graph,
task prompts, and hashes are staged in
`generated/CONTRACT_V2_PROMPT_REVIEW.md` for author approval. The method uses one
isolated workspace and two Harness calls per round: the Decider chooses ten
concrete CRNs and the Writer implements constraints and JSON encoding.
The launchers fail closed until
`generated/CONTRACT_V2_PROMPT_APPROVAL.json` explicitly approves the SHA-256 of
the current review JSON. The adjacent `.example.json` is deliberately
non-approving and is never promoted automatically.
They also require a frozen `generated/analysis_plan_v2.json`. Primary paper
campaigns additionally require a hash-matched passing
`generated/contract_v2_sentinel_report.json`; a sentinel launch must be named
explicitly with `--campaign-stage sentinel`.

## Artifact state

The strict analysis must reject missing primary artifacts, any contract version
below 2, a protocol-fingerprint mismatch, or accepted LLM candidates that reuse a
fixed-template reaction ID. The active tree intentionally contains no complete
primary LLM cohort. Audited RL-only controls remain available. Historical
classifier and oscillator-mean LLM runs are retained as replication evidence
because those tasks have no fixed template reactions, but the recovery plan calls
for a homogeneous version-2 primary rerun. Local models, RAG, and
optimization-exclusion remain unevaluated future directions.

The only permitted historical SI use is the registry-guarded request-0 HOF
shown/withheld diagnostic. It supports protocol selection only and must disclose
that the historical LLM validator did not forbid reuse of a fixed-template
reaction ID.
