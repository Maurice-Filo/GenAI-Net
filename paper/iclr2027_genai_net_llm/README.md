# Assessing Full-Duplex LLM–RL Communication via Imitation Learning for Biomolecular Network Design

Anonymous ICLR-format manuscript assessing simulator-in-the-loop communication
between RL and LLM search over chemical reaction networks. The complete method
withholds the random initial HOF, then can expose evaluated RL/HOF and
replay state from request 1 onward. Every LLM arm retains
evaluator outcomes for its own prior proposals. Archived runtime fields retain
the abbreviation SIL. At the loss definition, the manuscript calls replay
that can include exogenous proposals **Elite Imitation Training (EIT)**; its
mathematical form is inherited from SIL. The title and abstract frame an
empirical study of communication, and the fixed 100-epoch horizon characterizes
ongoing search rather than establishing converged performance.

## Evidence state

The main-text communication results use explicit audited contract-v2 registries:

- 160 DeepSeek V4 Flash full-duplex runs across eight tasks;
- 60 matched Flash no-communication runs on RPA, Logic, and classifier;
- 40 Flash Logic directional runs, completing its four-arm design; and
- a local Qwen3.8-27B-DFlash2 $2\times2$ directional communication study on RPA
  and Logic, with 20 complete Logic seeds and 17 complete RPA cases.

The Logic interventions support LLM-to-RL insertion relative to uncoupled search,
but do not detect an incremental loss or successful-yield benefit from adding
RL/HOF-to-LLM conditioning over insertion alone. A separate main-text post-hoc
analysis finds that return conditioning broadens directly proposed LLM structures,
with corrected topology-count evidence on local Logic. It covers 308 completed
audited databases, excludes RL proposals and the terminal top-30 cap, and corrects
all 24 exploratory proposal-distribution tests. This is not performance synergy.
The four storage-invalid RPA arm--seed records remain excluded and disclosed.

All 1,139 qualifying Logic topology-by-seed records pass every truth-table row
at the evaluated horizon. Exhaustive output-preserving pruning leaves mean
insertion versus independent-union counts of 14.35 versus 4.65 for Flash and
7.10 versus 1.00 locally (both post-hoc Holm p=0.00304). This diagnostic checks
removable reactions at fixed surviving rates, not general mechanistic equivalence
or biological implementability.

The final appendix, **Genetic Search and Parameter-Optimization Comparisons**,
adds a separate 60-run, no-LLM RPA workflow comparison (AdaGrad hybrid,
evolution-fit hybrid, standalone GA). It distinguishes the planned four-test
nominal endpoint family from a post-hoc 15-test perturbation family. The latter
supports higher useful perturbed-solution yield for AdaGrad versus GA, not
corrected significance versus evolution-fit or a pure robustness claim.
It also includes state-ceiling diagnostics and the balance-preserving control.
These recipes differ from the communication cohorts and do not replace their
missing same-proposal causal controls. All supporting counts and p-values are
in `generated/rpa_optimizer_comparison.json` and
`generated/rpa_optimizer_perturbation_tests.csv`.

Regenerate this appendix's tables and two paper-sized figures from the repo root:

```bash
.venv/bin/python -m paper.iclr2027_genai_net_llm.generate_ga_optimizer_appendix \
  /local1/rossin/search-exploration/rpa-statistical-20seed-20260915 \
  /local1/rossin/search-exploration/rpa-rate-perturbations-20260917
```

## Regenerate and build

```bash
MPLCONFIGDIR=/local0/tmp/mpl python3 analyze_contract_v2_primary.py
MPLCONFIGDIR=/local0/tmp/mpl python3 generate_submission_assets.py
make
```

Run `make amplification` to regenerate the source-aware portfolio figures for
local Logic and Flash RPA/classifier. Definitions and current results are in
`LOCAL_LOGIC_AMPLIFICATION.md` and `FLASH_PORTFOLIO_AMPLIFICATION.md`.
The same target also regenerates threshold/cap sensitivity and request-opportunity
accounting in `PORTFOLIO_SENSITIVITY_OPPORTUNITIES.md`.
`PORTFOLIO_FIGURE_PLAN.md` maps the main/SI figures and scope.
Run `make proposal-diversity` to regenerate the main/SI proposal-distribution
figures and complete tables from the audited databases. Definitions and results
are in `LLM_PROPOSAL_DIVERSITY.md`.
Run `make posthoc-checks` to verify frozen databases and saved request contexts,
recompute the separate 22/4/6-test post-hoc families, and regenerate their SI
tables and `generated/posthoc_checks.csv`. Source data fingerprints and detailed
results are in `reviews/repo_answers_2026-09-09/results.json`. This target makes
no model calls, search runs, or new simulations.

Run `make logic-pruning` to replay all 4,800 terminal Logic records, check all
16 truth-table rows, and exhaustively evaluate the 32 reaction subsets of each
qualifying, correct design. The fixed template and surviving rates are preserved.
The post-hoc rules are frozen in `generated/logic_correctness_pruning_plan.json`;
the primary maximum OUT deviation is 0.01 over the original 16-by-1,000 sampled
trajectories, with descriptive tolerances 0.001 and 0.05. Per-run artifacts permit
resumption and preserve every subset outcome. The target verifies the original
database hashes and published qualifying counts before rendering appendix tables.
It uses new simulations but no model calls or search runs. Results are written to
`LOGIC_CORRECTNESS_PRUNING.md` and `generated/logic_correctness_pruning*.json/csv`.

The build runs `audit_contract_v2_readiness.py` before `pdflatex`. The compiled
paper is `build/main.pdf`. The conclusion ends on page 9; required statements
follow before the references, subject to final venue policy. Section 4.3 and Figure 3
present proposal diversification; Appendix F provides the complete analysis.
Post-hoc endpoint correction is in Appendix E.1, exposed-HOF overlap in F.1,
supplied-case classifier correctness in J.2, and Logic correctness/pruning in J.3.

Run focused tests with the checkout first on the import path:

```bash
PYTHONPATH=../.. MPLCONFIGDIR=/local0/tmp/mpl ../../.venv/bin/pytest -q \
  ../../tests/test_hybrid_runner_configuration.py \
  ../../tests/test_llm_integration.py
```

## Important analysis constraints

- Portfolio qualification is strict loss below the task threshold frozen from
  the median RL-only endpoint.
- Final topology identity is invariant to reaction serialization order and rate
  values, but preserves labelled species, reaction identity, and input channels.
- The available RL-only records have no terminal top-30 archives. The frozen
  eight-task RL-only portfolio family is therefore not estimable and is not
  reconstructed or imputed.
- The hosted communication evaluation uses the separately prespecified, matched
  full-duplex versus no-communication family on RPA, Logic, and classifier.
- Directional tables retain raw paired tests; a separate post-hoc correction
  covers the 22 reported Logic endpoint tests. Additional separate post-hoc Holm
  families contain four HOF-overlap contrasts and six classifier correctness
  contrasts. Logic correctness and pruning use another four-test post-hoc family
  (two models by two criteria, insertion versus independent union).
  The direct-proposal analysis retains its 24-test family. None is a
  new confirmatory or exhaustive study-wide family.
- Emitter labels and chronological topology relationships are mechanism
  diagnostics, not standalone causal ancestry.

## Structure

The reader sees the implementation motivation and empirical question first,
followed by Table 1 with all four communication arms and both models' Logic
yields. Study design and the independent-union control precede the inherited
update. Results progress from replicated insertion gains to classifier
functional support, return-channel breadth, and the eight-task RL-only context.
Related work follows results. The architecture and combined rank-one overview
are in the appendix; the classifier portfolio figure is in the main text.
The statistical families and their exploratory or prespecified status are
unchanged by this presentation order.
The original architecture artwork is retained without altering its embedded labels.

An optional independent search baseline is available in
`../../comparisons/llm_seeded_evolution/`. It replays only proposals from the
**No communication** branch, at their original completion boundaries, and
uses evolutionary reaction/rate mutations with no new model calls.

`compare_search_algorithms.py --methods no_communication llm_to_rl llm_seeded_evolution`
adds it to an optional comparison table after full-budget runs complete.
Other available choices are `rl_to_llm` and `full`. Pilots are excluded.
The existing communication-arm endpoints use different proposal streams and
are labelled as contextual comparisons; only No communication shares the
baseline's exact proposals. These outputs are separate from the frozen
confirmatory families and do not automatically alter the manuscript.

- `main.tex` and `sections/`: anonymous paper and supplementary material.
- `analyze_contract_v2_primary.py`: audited rank-one, portfolio, and trajectory
  analysis.
- `generate_submission_assets.py`: central figures, provenance/local analysis,
  machine-readable results, and LaTeX tables.
- `generated/submission_results.json`: numerical source for the active narrative.
- `audit_contract_v2_readiness.py`: fail-closed manuscript build gate.
- `figures/`: manuscript-ready PDF and review PNG figures.

The frozen prompt appendix is generated from the system, Decider, Writer, task,
and output-contract sources. Each LLM round uses one isolated
workspace and two Harness calls: the Decider selects ten concrete CRNs and the
Writer implements the constraints and encodes JSON. Each Writer member is
validated independently before canonical CVODE evaluation.
