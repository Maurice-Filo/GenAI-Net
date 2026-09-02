# Contract-v2 experiment recovery plan

Status: static proposal-space mask audit passed; blocked on author
prompt/evaluator approval, a frozen analysis plan, and dynamic sentinel
preflight. No experiment is currently running.

This plan replaces the quarantined campaign schedule. Its purpose is to produce
one coherent, auditable paper dataset rather than repair historical numbers in
place. The main scientific estimand is the amount of structurally distinct,
simulator-qualified design material retained by full-duplex search. Rank-one loss
is secondary, and diversity is always reported conditional on quality.

## 1. Evidence disposition

| Evidence | Disposition | Reason |
|---|---|---|
| RL-only, 100 epochs, all eight deterministic tasks | Reuse after strict audit | Native action mask was correct; candidate budget and solver are already matched |
| Historical classifier and oscillator-mean full duplex | Development-only; not manuscript evidence | They did not expose fixed template IDs, but they do not implement the complete v2 protocol and cannot support a paper claim |
| Historical request-0 HOF shown/withheld diagnostic | Quarantined; SI-only protocol-selection evidence | May support only the decision to withhold the random request-0 HOF; the LLM contract did not forbid reselecting fixed-template reaction IDs |
| Every other pre-fix LLM run | Quarantined; never use for claims | Model received the wrong proposal space even where no collision was realized |
| Historical full/no-communication portfolio analysis | Development only | Invalid proposal contract and unmatched terminal/cutoff semantics |
| Historical Pro, local, non-reasoning, unregistered initial-HOF, long-300, and prompt ablations | Development only | They inherit the invalid task contract when run on fixed-template tasks |
| Historical generated figures/tables/PDF | Quarantined | Derived from one or more invalid cohorts |

The new primary analysis must never scan the quarantine root. Artifact discovery
will use explicit campaign IDs and protocol fingerprints, not broad glob patterns.
The request-0 diagnostic is the sole exception: it must be read through an
explicit SI-only registry and analysis entry point that labels every output
`protocol-selection-only`. It cannot contribute a candidate, endpoint, CRN,
portfolio, communication, or model-performance value anywhere else in the paper.

## 2. Blocking issues and compensating actions

### P0. Fixed-template reaction mask

Problem: the old Harness library exposed reaction IDs that native RL had already
masked because they occur in the fixed template.

Action:

1. Keep the evaluator fix that unions `crn_template.gather_reaction_IDs()` with
   the null and configured forbidden IDs.
2. Set `proposal_space_contract_version=2` in every campaign, run, workspace,
   database, and completion manifest.
3. Record exact `template_reaction_ids`, `null_reaction_id`,
   `forbidden_reaction_ids`, allowed-library hash, and output-contract hash.
4. Reject a launch unless each fixed ID is absent from the model-facing TSV and
   JSON schema, present in the evaluator forbidden set, masked in the initial RL
   observation, and rejected before simulation by a deliberate negative test.
5. Keep the fixed-ID/input-channel collision regression test.

Pass gate: all deterministic tasks pass the four-way RL/evaluator/schema/library
mask check, and no accepted candidate can reuse a fixed ID.

### P0. Communication endpoint mismatch

Problem: the historical portfolio analysis combined a terminal 100-epoch
full-duplex archive with an epoch slice from a 300-epoch no-communication run.
The arms therefore did not share identical wait and snapshot semantics.

Action:

1. Run dedicated 100-epoch full and no-communication arms under one paired
   campaign for RPA, Logic, and classifier.
2. Persist two named boundaries: `training_boundary` after exactly 100 RL batches
   and before terminal waiting, and `terminal_boundary` after all scheduled
   requests have completed, failed, or timed out.
3. At the training boundary, include only responses merged by that boundary in
   both arms. At the terminal boundary, pool all accepted scheduled responses in
   both arms.
4. Never infer batch count from a snapshot label alone. Store cumulative RL,
   LLM-tool, LLM-final, and total evaluator counts on every snapshot.

Pass gate: paired arms have identical seeds, schedules, caps, timeout policy,
snapshot semantics, and candidate-accounting fields.

### P0. Explicit Decide-then-Write boundary

Problem: the old single request blurred scientific proposal and machine encoding,
and a malformed member could discard unrelated valid proposals.

Action: freeze an explicit **two-stage Decide-then-Write protocol** inside one
isolated Harness workspace. Stage 1 (Decider) selects ten concrete CRN designs,
including their intended rates. It may use reaction equations, species names,
library IDs, tables, or another concise scientific notation; it is not required
to emit JSON. It must not defer scientific choices to Stage 2. Stage 2 (Writer)
reads the Decider artifact and supplied library, applies the fixed constraints,
and encodes those designs in the exact JSON contract. The Writer is an encoder
and constraint implementer, not an independent proposal agent. Each stage is one
provider call through Harness; both calls and their edge are logged to Comet.
`DECIDER_DESIGNS.md` is a short scientific design record, not private chain of
thought. Automatic Writer retries are disabled for paper campaigns so every
scheduled round has exactly two provider calls.

Pass gate: five scheduled rounds produce ten provider calls per 100-epoch run;
each workspace contains the Decider request/response, Writer request/response,
member-level validation audit, clamp audit, and final accepted payload.

### P0. Stale manuscript evidence

Problem: the active TeX source contained historical numerical claims, including
invalid portfolio and source-filtered results.

Action: a forensic source copy is quarantined and the live manuscript now contains
method text plus explicit result placeholders. Keep builds blocked until v2
cohorts pass audit. Then regenerate every numeric table, figure, result sentence,
abstract claim, and conclusion statement from versioned outputs. Do not hand-edit
values into TeX. Do not restore the obsolete RPA Jaccard values or any use of
“clean Logic” for the pre-fix contract.

Pass gate: every manuscript number has an artifact key and generating script;
the strict build fails if a quarantined, v1, incomplete, or fingerprint-mismatched
run is referenced.

### P1. Prompt/evaluator agreement

Problem: the old Logic prompt described terminal BCE while the executable loss
is weighted trajectory L1. Breadth oscillator prompts also omit numerical feature
weights and some task details.

Action:

1. Generate prompts from a frozen typed task-contract record.
2. Use the trajectory-faithful Logic wording in the v2 primary cohort.
3. Freeze all scenarios, targets, initial conditions, horizon, time points,
   tolerances, transient weights, oscillation feature weights, reaction budget,
   and rate bounds in both machine-readable and human-readable form.
4. Hash the task contract, rendered prompt, evaluator source files, and config.
5. Optionally run the historical BCE wording as a matched SI sensitivity arm;
   it cannot define the primary method.

Pass gate: a field-by-field generated report shows no disagreement between each
prompt and executable evaluator.

### P1. Reaction equality, ordering, and structural distance

Problem: numeric reaction-ID sets erased the distinction between an
input-modulated fixed reaction and an unmodulated added reaction sharing its ID.

Action: keep topology identity as a sorted set of complete labelled structural
records: library ID, reaction implementation type, reactants, products, and input
channels. Exclude kinetic parameters and serialization order. Subtract exact
fixed-template records, never numeric IDs, before structural-distance analysis.
Preserve RL construction order only for SIL replay.

Pass gate: tests prove that reaction-order permutations and parameter retuning
share a topology hash, while input-channel or labelled-structure changes do not.

### P1. Asynchronous communication and insertion opportunity

Problem: model latency changes how many responses can influence RL before the
training boundary. Provider queueing and CPU contention can confound a simple
epoch plot.

Action: log scheduled, launched, service-started, completed, validated, merged,
failed, timed-out, capacity-skipped, and terminal-only requests with wall-clock
times and RL epochs. Continue RL while calls wait. Report the number of insertion
opportunities and the fraction of accepted proposals arriving before the training
boundary. Counterbalance communication arms in the same resource blocks.

Pass gate: a deliberately slow fake model demonstrates uninterrupted RL progress,
and all request accounting reconciles across logs, SQLite, and completion files.

### P1. Candidate budget and response handling

Problem: the hybrid has a conservative 50-candidate budget disadvantage, and an
invalid member must not discard unrelated valid proposals.

Action: validate each returned candidate independently for both hosted and local
models. Preserve every valid member, reject each invalid member with an indexed
reason, and do not synthesize replacements. Exact duplicate candidates are
rejected after the first valid occurrence. Count only actual canonical evaluator
calls. At 100 epochs the control evaluates 102,400 RL candidates; hybrid arms
evaluate 102,300 RL plus at most 50 accepted LLM candidates. Tool evaluations,
if enabled in a future study, consume the same cap. Log requested, returned,
accepted, rejected, clamped, evaluated, and merged counts separately.

Pass gate: cumulative counts never exceed the cap; one malformed member cannot
remove another valid member; and every unevaluated member has a durable reason.

### P1. Parameter support asymmetry

Problem: the previous LLM interval `[0.1, 50]` was narrower than intended, while
the shared RL stream has positive unbounded support.

Action: set the contract-v2 LLM interval to `[0.001, 100]`. Inform the Decider
that a proposed value below 0.001 or above 100 is deterministically truncated to
0.001 or 100, respectively. Require the Writer to perform that translation and
enforce it again in the host before member validation. Every host-side clamp is
recorded with candidate, reaction position, original value, and final value.
Non-numeric and non-finite values are rejected, not repaired. State that the
hybrid still contains the same positive-unbounded RL support as RL-only and do
not claim symmetric proposal distributions.

Pass gate: the prompt, schema, evaluator, manifest, and clamp audit agree; no
direct LLM value outside `[0.001, 100]` reaches simulation.

### P1. Provenance and causal language

Problem: “RL provenance” identifies the emitter, not whether an RL-emitted record
reuses and retunes a topology previously introduced by the LLM.

Action: store `emitter=RL|LLM` separately from `provenance_class`. Use the classes
`direct_llm`, `rl_native_topology`, `rl_exact_reemission_of_llm_candidate`, and
`rl_parameter_refinement_of_llm_topology`. For the latter two, persist the prior
LLM proposal reference and first-seen epoch. These labels establish chronology
and exact topology/parameter relationships, not a counterfactual cause: only a
communication intervention supports a behavioral causal claim. Expose both
fields in SQLite, generated tables, and the viewer.

Pass gate: every retained HOF member has an emitter and provenance class; related
LLM references resolve; and no text attributes a later RL candidate solely to
SIL or LLM ancestry without a dedicated intervention.

### P1. Post-hoc portfolio estimand and statistics

Problem: the old central metric was formulated after inspecting results, and
multiple threshold views could invite selective reporting.

Action before launching:

1. Freeze numeric task thresholds from the already independent RL-only controls.
2. Define the primary portfolio as the number of distinct labelled topology
   hashes among the final top 30 candidates with canonical loss strictly below
   the frozen threshold.
3. Define the threshold sweep, qualified-candidate count, within-qualified exact
   structural Jaccard distance, and source-filtered archive as secondary.
4. Freeze tie tolerance, failed/missing-run policy, and all exclusions.
5. For eight rank-one task tests report all 20 pairs, median and IQR, W/T/L,
   median paired ratio, two-sided Wilcoxon, and Holm correction.
6. Treat RPA, Logic, and classifier portfolio tests as one named family and
   adjust them together. Report effect sizes and raw paired data regardless of
   significance.

Pass gate: an immutable analysis-plan JSON and hash exist before the first v2
model request.

### P1. Reproducibility and observability

Problem: old run manifests do not uniformly contain source commits, prompt
hashes, inference mode, DSH version, or an authoritative Comet status. Hosted
model weights are not reconstructable from a model label.

Action: store repository commit plus dirty-diff/source-bundle hash, Python and
package lock, CUDA/GPU/CPU details, exact DSH package/version/profile patch,
provider/model identifier, request and response hashes, token/cost metadata when
available, system/task/contract/library hashes, and all evaluator files. A run
may be skipped only when its full protocol fingerprint matches. SQLite and local
manifests are authoritative. Comet receives mirrored progress only after a
one-event authentication smoke test. Every task/run must create a named Comet
experiment and mirror campaign metadata, RL progress, both model stages, member
validation, clamping, latency, provenance counts, and completion state. Launch is
blocked unless the smoke succeeds; a later Comet outage is recorded explicitly
and cannot be described as successful logging.

Pass gate: a run can be reconstructed and audited without Comet or mutable DSH
profiles, except for the hosted model weights explicitly identified as external.

### P2. Deferred mechanisms

Local models, RAG, optimization-exclusion, stochastic RPA, and habituation are
not part of the corrected paper evidence. They remain future directions until
they each receive a separate frozen protocol, budget accounting, and matched
control. This prevents optional machinery from delaying or obscuring the core
full-duplex result.

## 3. Frozen version-2 protocol

- Tasks: RPA, Logic, classifier, Hill, ultrasensitive, biphasic, oscillator mean,
  and oscillator frequency; deterministic CVODE only.
- Seeds: 0 through 19, paired by RL seed.
- Primary horizon: 100 RL batches.
- RL batch: 1,023 for hybrid; 1,024 for RL-only.
- Candidate cap: 102,400.
- HOF capacity: 30 distinct complete-record labelled topologies, best parameters
  retained per topology.
- LLM schedule: epochs 0, 20, 40, 60, and 80.
- LLM request: one Harness workspace, one Decider call and one Writer call, ten
  concrete Decider designs and up to ten independently valid encoded candidates.
- Communication: request 0 receives no HOF; later requests receive HOF, SIL
  status, prior evaluator evidence, and exclusions (empty in the primary study).
- Tools: zero simulations, zero literature searches, no RAG, and no dynamic
  exclusion/optimization.
- Validation: independent members; exact reaction budget, unique allowed IDs,
  correct parameter arity, finite values clamped to `[0.001, 100]`, no exact
  duplicate. Returning fewer than ten valid members is recorded, not hidden.
- Execution: LLM requests are asynchronous and never block RL updates.
- Hosted model: exact logged DeepSeek V4 Flash provider identifier; provider
  default reasoning mode, explicitly labeled rather than inferred.

Task-specific CVODE tolerances and every objective coefficient are part of the
frozen task-contract JSON, not assumed to be uniform across tasks.

## 4. Campaign schedule

### Gate A: static and short dynamic preflight

Run all unit/integration tests, the eight-task mask/evaluator contract audit, a
secret-path scan, a one-event Comet smoke, and a short non-paper sentinel campaign.
Inspect every generated contract before releasing seed 0. Sentinel output is
stored outside paper methods and can never satisfy a primary artifact lookup.

### Cohort B: 100-epoch primary and communication study

Run 20 full-duplex Flash seeds for all eight tasks: 160 runs and at most 1,600
model calls (800 Decide-then-Write rounds). For RPA, Logic, and classifier,
launch the matched no-communication arms in counterbalanced resource blocks: 60
additional runs and at most 600 model calls. The full-duplex runs from those
blocks also serve the primary endpoint comparison; they are not duplicated.

Reuse the 160 audited 100-epoch RL-only controls. Their frozen endpoints also
define the quality thresholds before v2 outcomes are inspected.

Suggested IDs:

- `flash-v2-primary-full-100epoch-20seed`
- `flash-v2-communication-none-100epoch-20seed`
- method roots ending in `_contract_v2`; never reuse a historical method name or
  suffix.

Release rule: run one final seed per task/arm, audit it, then release seeds 1--19.
Do not continue a task if any protocol fingerprint, mask, budget, database, or
request-accounting check fails.

### Cohort C: corrected 300-epoch horizon

After Cohort B is locked, run RPA and Logic for 20 seeds in three arms: full
duplex, no communication, and RL-only. This is 120 runs. Hybrid arms use 300 x
1,023 RL candidates plus at most 15 x 10 LLM candidates (maximum 307,050) and at
most 30 model calls per run; RL-only uses 300 x 1,024 = 307,200. Analyze the
300-batch terminal endpoint and explicitly named intermediate training boundaries
without treating snapshot labels as batch counts.

### Cohort D: sensitivity analyses, only after the main freeze

- Logic prompt conditioning: use the v2 trajectory prompt as the primary arm and
  run 20 matched seeds with the historical BCE wording if this SI diagnostic is
  still useful.
- Request-0 context: the quarantined shown/withheld comparison may appear only as
  an SI protocol-selection diagnostic. Disclose that its LLM validator did not
  forbid fixed-template reaction IDs. Do not rerun it unless a stronger claim is
  later required, and never use it for CRN quality or method performance.
- Model sensitivity: rerun Pro on corrected RPA and Logic only if a model-capacity
  claim remains in the paper. Historical Pro results cannot be reused.
- Non-reasoning/minimal client: rerun only if retained as a claim; otherwise omit.

No local, RAG, exclusion, SSA, or habituation campaign is scheduled here.

## 5. Resource and concurrency policy

Use a 64-core host token pool. Allocate 60 numerical worker slots (15 runs x four
CVODE workers) and reserve four cores for orchestration, SQLite writers, Harness
processes, and plotting; this uses the processor without starving the asynchronous
communication path. Increase to 16 x four only if a measured preflight shows no
queue starvation or throughput regression. Set BLAS/OpenMP thread counts to one.

Hosted-model campaigns do not reserve a GPU for local inference. Select the
faster available GPU for RL in a short frozen throughput test and keep that
assignment fixed within each paired campaign. Cap global provider concurrency at
eight and log slot queue time. Counterbalance full/no-communication jobs across
the same blocks so provider latency and machine load are not confounded with arm.

Never overlap local-model inference with these primary campaigns. No campaign
may resume from a pre-v2 checkpoint or write into a historical output root.

## 6. Analysis outputs

Produce one versioned analysis directory containing:

1. Per-seed endpoint CSV with losses, candidate counts, protocol fingerprint,
   completion state, and all request counts.
2. Quality-conditioned portfolio CSV at the frozen threshold and prespecified
   sensitivity thresholds.
3. Best-loss and qualified-topology trajectories using actual cumulative
   evaluator counts on the x-axis.
4. Insertion-opportunity table: launched, served, valid, merged before training
   boundary, terminal-only, failed, timed out, and capacity-skipped.
5. Emitter and chronology-based provenance diagnostics, distinguishing direct
   LLM records, native RL topologies, exact re-emissions, and parameter refinements;
   these are not by themselves causal ancestry estimates.
6. Exact structural-distance results using complete records and fixed-record
   subtraction.
7. Paired raw points, W/T/L, medians/IQR, ratios, Wilcoxon statistics, and named
   multiplicity families.
8. Cost/latency table from local process logs, with Comet used only as a mirror.

All plots use consistent colors: RL blue, full duplex green, direct LLM/original
context orange, and independent/no-communication gray. Loss axes use log scale
where positive values span orders of magnitude. Diversity panels always show or
condition on performance.

## 7. Manuscript rewrite checklist

1. Rewrite the abstract around qualified portfolio yield, then state rank-one
   effects as task-dependent secondary evidence.
2. Describe the implemented SIL equation exactly as code and full duplex as the
   complete HOF/SIL/evaluator communication path, without component-level causal
   claims.
3. Describe the two-stage Harness protocol exactly: the Decider owns scientific
   CRN choices and the Writer implements constraints and JSON encoding.
4. Replace all historical result prose and regenerate every numerical asset.
5. Put prompt-conditioning, strictly scoped historical initial-HOF selection,
   member rejection, rate clamping/support, provider reproducibility, and the
   contract-mask discovery in the SI.
6. Report non-improving tasks without hiding them; interpret proposal quality and
   kinetic-tuning difficulty only as diagnostics supported by request-level data.
7. Present the amount of good solutions as the primary result and structural
   separation as a qualified secondary property.
8. Keep local inference, RAG, and optimization-exclusion as future directions.
9. Run a final data-to-text audit, anonymous-format check, nine-page check, and
   clean-room build from a frozen source bundle.

## 8. Tomorrow's start sequence

1. Confirm no workers are active and inspect GPU/CPU health.
2. Implement and test the protocol-v2 preflight, member validation, clamping,
   provenance, Comet, and fingerprint fields.
3. Freeze `analysis_plan_v2.json`, including numeric RL-derived thresholds and
   statistical families; record its SHA-256. The checked-in
   `analysis_plan_v2.DRAFT.json` is non-releasable until all pending fields are
   resolved and its status is explicitly changed to `frozen`.
4. Generate the shared system prompt, both role-specific Harness wrappers, all
   eight task prompts, Decider and Writer templates, and output contracts, then
   stop for explicit author approval.
5. Require `generated/CONTRACT_V2_PROMPT_APPROVAL.json` to carry the exact
   SHA-256 of the current review JSON plus non-empty author and timestamp fields.
   Both the campaign launcher and every worker fail closed when it is absent,
   stale, or not explicitly marked `approved`.
6. Only after prompt approval and analysis-plan freezing, run the short sentinel
   campaign with the explicit `sentinel` release stage and reconcile SQLite,
   workspace, Comet, and counters.
7. Run and audit seed 0 for each released arm.
8. Freeze a sentinel report carrying the exact prompt-review and analysis-plan
   hashes. Only after every gate is green may the `paper` release stage launch
   the remaining paired campaign blocks and restart the database dashboard on
   the new v2 campaign plan.

Any failure pauses only the affected task/arm. Diagnose from preserved artifacts,
make the smallest code change, bump the protocol fingerprint, invalidate the
sentinel, and repeat the gate. Never patch a scientific result in place.
