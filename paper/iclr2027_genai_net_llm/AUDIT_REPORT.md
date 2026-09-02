# Paper and Experiment Audit

Audit date: 1 September 2026

## Supersession status

This report records the forensic audit of the historical, pre-template-mask
campaigns. It is not a validation of an active paper cohort. After the audit,
every pre-fix LLM run on a task with one or more fixed template reactions and all
dependent numerical assets were moved to the quarantine documented in
`PRE_MASK_QUARANTINE.md`. The earlier interpretation that Logic was “clean”
because no stored proposal realized the collision was too narrow: its model-facing
proposal space was still wrong. `EXPERIMENT_RECOVERY_PLAN.md` defines the only
admissible path to new paper results.

## Scope

- Recomputed all 160 matched task/seed endpoint pairs from frozen artifacts.
- Cross-checked all 160 hybrid endpoint files against immutable SQLite HOF snapshots.
- Checked 160 RL controls and 160 hybrid runs for seed, CVODE solver, epochs,
  batch size, candidate accounting, reaction count, and duplicate reactions.
- Audited all 800 scheduled primary LLM requests and their run logs.
- Reconstructed the matched epoch-100 full-duplex and no-communication elite
  archives for the quality-conditioned topology analysis.
- Checked every accepted LLM candidate and every final HOF entry for reuse of a
  reaction ID already masked by the fixed task template.
- Recomputed fixed-template-excluded reaction-set distance and source-filtered
  RL-emitter diagnostics from all RPA and logic HOF snapshots through epoch 100.
- Re-ran the repository test suite against this checkout.

The machine-readable forensic audit remains in
`generated/paper_experiment_audit.json`. Historical communication summaries and
per-seed tables were moved with the invalid derived evidence. The complete move
inventory is `generated/pre_mask_quarantine_inventory.json`.

## Material findings

1. **Logic prompt/evaluator wording.** The executed prompt described the logic
   task through final-state binary cross entropy, while `LogicTaskKind` invokes
   transient-weighted trajectory L1 over all 16 truth-table rows. RL and hybrid
   used the same canonical evaluator, so their numerical comparison is internally
   controlled. Both objectives encode the same truth table, and in practice the
   logic loss is dominated by steady-state mismatch. The exact executed prompt is
   disclosed in the supplement.

2. **Parameter-domain asymmetry.** LLM payloads are restricted to `[0.1, 50]`.
   Native RL uses a positive log-normal policy without that lower bound. Final
   rates below 0.1 occur especially often for Hill (9/20 hybrid and 13/20 control
   endpoints). The LLM floor intentionally prevents satisfying a fixed reaction
   budget by effectively deleting reactions with zero or negligible rates. Its
   disclosed tradeoff is that the agent cannot directly propose slower kinetics.

3. **Atomic response rejection.** One invalid candidate rejects all ten members
   of a payload. Across the final policy, 779/800 batches were accepted. Twenty
   rejected batches contained an out-of-range parameter and one contained an
   exact within-batch duplicate. The hybrid consequently used no more, and often
   fewer, numerical evaluations than reported by its scheduled maximum. This is
   an implementation limitation, not a reason to rerun the completed campaign.

4. **Historical Harness validation omitted the template mask.** Native RL masks
   every reaction ID already used by the fixed template. The historical LLM
   evaluator instead rejected only the null reaction and within-candidate
   duplicates. It therefore accepted 497 RPA candidates that reused fixed ID 1
   or 28; 134 such records remain in final RPA HOF snapshots and 9/20 rank-one
   endpoints are affected. The same audit finds one accepted biphasic proposal
   and 133 oscillator-frequency proposals, but no affected rank-one endpoint in
   either task. Logic, Hill, ultrasensitive, and some other cohorts have zero
   *realized* collisions, but this does not make their protocol valid: every task
   with a fixed template exposed the wrong proposal library and must be rerun.
   Only classifier and oscillator mean have no fixed template IDs and are
   unaffected by this defect. This is a proposal-space asymmetry, not a
   topology-hash ambiguity. The evaluator and Harness contract now forbid all
   template IDs, and strict paper finalization rejects all affected historical
   artifacts.

5. **Request-0 context remains a restricted protocol diagnostic.** A paired
   shown/withheld comparison informed the decision to omit the random initial HOF.
   Its LLM validator did not forbid fixed-template reaction IDs. The artifacts
   therefore remain quarantined and may support only that protocol-selection
   decision through the explicit SI registry. They cannot support candidate,
   endpoint, portfolio, communication, or model-performance claims.

6. **Provenance denotes the emitting process, not ancestry.** Analyses identify an
   LLM-provenance candidate by exact topology hash plus serialized parameters; all
   optimizer-emitted candidates have RL provenance. An RL-provenance candidate may
   refine an LLM-discovered topology through the shared HOF. HOF records better
   than the current RL batch are eligible for the implemented SIL loss, but logs
   do not prove that one replay caused a later candidate. Isolated RL supplies the
   matched behavioral comparator for the complete communication pathway.

7. **Inference mode was not uniformly frozen.** Logic and breadth manifests say
   `provider-default`; the earlier RPA manifest omits the field. The complete
   Harness versus minimal non-reasoning comparison is a compound agent ablation,
   not a clean hidden-reasoning ablation.

8. **Final-policy RPA budget metadata is stale.** Twenty 100-epoch RPA artifacts retain
   `full307200`/`budget_cap=307200`, but their configurations, progress, databases,
   and counters agree on 100 epochs, 102,300 RL candidates, and at most 50 LLM
   candidates. Raw metadata remains immutable and the paper discloses the label.
   Separate HOF-exposed and no-communication RPA/Logic runs completed 300 epochs,
   but the HOF-exposed policy is not the defined method and is not used for a main
   claim. The epoch-100 independent pool is reconstructed only from no-communication
   candidates available by that epoch.

9. **The historical portfolio metric is post hoc and its arm cutoffs were not
   identical.** Among each method's 30 best nominal epoch-100
   candidates, we count distinct topology hashes below an independently defined
   RL-only loss threshold. At the median matched RL-only endpoint, full duplex
   yields 21.5 versus 13.0 qualifying topologies on RPA and 8.5 versus 1.5 on
   logic. Both values are historical diagnostics only. Besides the invalid
   proposal contract, the analysis combined a terminal 100-epoch full-duplex HOF
   with an epoch slice reconstructed from a 300-epoch no-communication run; the
   response-wait and batch-boundary semantics were therefore not identical. The
   v2 study requires dedicated matched 100-epoch arms and a portfolio estimand
   frozen before outcomes are inspected.

10. **Optional extensions were disabled.** Primary completion records contain zero
    optimization-exclusion evaluations. Agent-triggered simulation and retrieval
    were also configured to zero. Local-model attempts are incomplete and are not
    analyzed as a controlled comparison.

11. **Topology identity is order-stable but label preserving.** The database hash
    contains reaction identity, implementation type, labelled reactants/products,
    and input channels. Reaction records are sorted before hashing and kinetic
    parameters are excluded. Thus serialization-order permutations and parameter
    retuning do not increase topology yield. Arbitrary graph isomorphisms are not
    collapsed because species labels, reaction identities, and the ordered pickup
    trajectory define task roles and the RL action/replay representation.

12. **Historical source filtering appeared to reject an append-only explanation.** Removing every exact
   direct LLM topology--parameter return from HOF history leaves median best loss
   `0.03583` versus `0.04793` on logic (20/20, `p=1.91e-6`), with qualified-
   topology medians 8 versus 0 (`p=2.72e-4`). The analogous historical RPA values
   (`0.00593` versus `0.00841`, and 10 versus 4) are contaminated by template-ID
   reuse and are not admissible evidence. Reaction-order-canonical matching changes
   zero source classifications, and the no-communication HOF contains zero direct
   LLM candidates through epoch 100. These values no longer establish a paper
   result because Logic used the wrong model-facing proposal contract and the
   communication comparator had unmatched cutoff semantics. At most, the method
   illustrates the intended emitter analysis: even a valid repetition would show
   changed RL emissions, not SIL-specific causality.

13. **RPA reaction-ID Jaccard also erased dynamically distinct records.** The old
    distance code reduced each topology to reaction IDs and subtracted fixed IDs
    `{1,28}`. In RPA this erased an inadmissible added unmodulated ID-28 reaction
    whenever it coexisted with the fixed input-modulated ID-28 reaction. Exact
    structural-record subtraction gives historical medians `0.5382` full duplex
    versus `0.5804` independent (18 paired seeds, 4/1/13 W/T/L, `p=0.0342`). Logic
    remains `0.4972/0.5565` (9 paired seeds, `p=0.3008`). The corrected RPA
    calculation is mathematically descriptive of the stored CRNs but still cannot
    be reported as protocol-valid evidence because those CRNs should have been
    rejected at proposal time.

## Why several endpoints remain near parity

Within the quarantined historical data, the observed pattern outside RPA was more
consistent with proposal quality than excessive concurrency. Median
best-batch/HOF ratios are below one for classifier (0.937), but
above one for Hill (1.105), ultrasensitive (1.195), biphasic (1.107), oscillator
mean (1.020), and oscillator frequency (1.046). Dose-response requests merge in a
median of three epochs. They simply seldom improve the current HOF. Historical
RPA request-quality metrics are retained only as diagnostics until rerun.

This remains a hypothesis for prompt and method design, not paper evidence. The
historical pattern suggested task fit plus implementation friction:
motif-rich adaptation and classification benefit from direct structural priors;
dose shaping and frequency control require precise kinetic tuning. The LLM lower
rate bound, under-specified oscillator feature weights, and atomic batch rejection
make that tuning harder. None alone explains parity, because 94--99 batches per
task still pass validation.

## Residual risks

- Legacy RPA/Logic RL controls contain configurations and outputs but no immutable
  source-commit identifier, so evaluator identity is supported by configuration
  and current code paths rather than a stored executable checksum.
- Hosted model weights cannot be reconstructed from the model name.
- Prompt/evaluator agreement is now a preflight requirement. A historical Logic
  wording arm and richer oscillator wording may be run only as named sensitivity
  analyses after the evaluator-faithful primary cohort is frozen.
- The eight endpoint tests were not preregistered; Holm correction is valid for
  the reported family but does not remove RPA policy-selection bias.
- The historical quality-conditioned portfolio was defined post campaign and
  used unmatched arm cutoffs. Its thresholds and statistical family must be
  frozen before a matched contract-v2 replication.
- A forensic copy of the historical manuscript source is quarantined. The active
  TeX now contains method text and explicit result placeholders; numerical assets
  and compiled PDFs are absent. No draft build is submission-ready, and all
  result text must be generated from v2 artifacts.

## Verification

```bash
PYTHONPATH=/local0/rossin/git/GenAI-Net \
MPLCONFIGDIR=/local0/tmp/mpl \
.venv/bin/pytest -q
```

The pre-quarantine full suite recorded 82 passing tests, including validator,
Harness-contract, template-mask, structural-record, and fixed-ID/input-channel
collision regressions. Focused post-quarantine verification records 23 passing
quarantine and LLM-integration tests. The strict artifact audit and manuscript
build now fail by design because invalid numerical inputs were removed. A full
contract-v2 suite and clean paper build are gates in the recovery plan.
