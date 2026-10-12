TASK: Evidence-guided design of five-reaction logic networks

Objective
Propose low-loss, structurally distinct mass-action CRNs implementing
OUT = (u_1 AND u_2) OR (u_2 AND u_3) OR (u_3 AND u_4).
Use evaluated workspace evidence to refine promising designs and explore alternatives.

Executable benchmark
- Inputs: four binary signals u_1, u_2, u_3, u_4, constant within each scenario.
- Species: X_1, X_2, X_3, X_4, OUT. There are no support species.
- Fixed template: empty -> X_i with propensity u_i, for i = 1..4.
  There are no fixed dilution reactions.
- Evaluate all 16 combinations in {0, 1}^4, starting every species at 0.01.
- CVODE: t = 0..100, 1000 time points, rtol = atol = 1e-8.
- Target throughout each trajectory: OUT = 1 on true rows and OUT = 0 on false rows.
  Seek these numerical concentrations, not merely qualitative high/low separation.
- Loss: mean weighted absolute output error over all 16 rows and 1000 times.
  Weights: 0.25 for the first 200 samples, 1 for the middle 600, 2 for the final 200.
  Divide the summed weighted errors by 16*1000, not by the sum of weights.
  Lower is better. Transient deviations count; correct final labels alone do not
  establish low trajectory loss. This is not a final-state cross-entropy objective.
- The evaluator stops trajectories reaching its all-state ceiling of 10,000.
  Low output loss does not establish bounded internal states or stability.

Design constraints
Choose exactly five distinct allowed added reactions from the supplied order-2
mass-action library. Keep the four fixed template reactions unchanged.
Use only the supplied species; do not select null reactions or template-reserved IDs.
Supply one finite scalar rate per added reaction. Rates below 0.001 are clamped
to 0.001; rates above 100 are clamped to 100. Account for this when designing.
DESIGN_CONSTRAINTS.md and the supplied library define admissibility.

Evidence-guided Decider policy
Read the current HOF overview when supplied. Further inspection of evaluated
candidates, prior LLM outcomes, diagnostics and plots is optional.
Choose promising parents by evaluated quality, not by RL or LLM origin.
Request 0 intentionally has no initial HOF: explore without recovering withheld
information. In any round, use only the feedback actually supplied in the workspace.

For ten requested candidates, aim for six evidence-guided refinements and four
structurally different exploratory designs. This is guidance, not a validity quota.
Without usable parents, explore instead. Spread refinements across promising parents.
Use observed limitations to motivate rate changes, reaction replacements or compatible
mechanism combinations. Specify each resulting CRN fully within the five-reaction budget.
Do not fill the batch with trivial rate variants or identical candidates.
Respect explicit topology exclusions.

The host may allocate AdaGrad rate fitting to selected structures. Provide plausible
complete rates nevertheless: fitting is limited and neither selection for fitting nor
improvement is guaranteed. Distinguish supplied-rate results from fitted results.
For each proposal, briefly identify its parent or evidence source, intended change and
uncertainty. Aggregate loss alone cannot identify a particular truth-table row failure.
Do not attribute improvement to one edit when several changes were made together.

The target detects at least one adjacent active pair in the chain 1-2-3-4.
Input count alone is insufficient: 1100 is true, while 1010 is false.
Check both false-row leakage and true-row output when cached diagnostics allow.
Do not invent diagnostics or introduce an unreported scoring objective.

Submit the batch for external evaluation and use returned evidence in later rounds.
Do not request new evaluations to pre-screen proposals. The Decider specifies complete
structures and intended rates in readable notation; the Writer encodes them without
changing scientific choices. Follow the shared system instructions for short notes,
JSON output, rate clamping and independent rejection of invalid members.
