TASK: Evidence-guided design of RPA reaction networks

Objective
Propose low-loss, structurally distinct mass-action CRNs that make X_3
track the reference u_1 while rejecting changes in disturbance u_2.
Use evaluated workspace evidence to refine promising designs and explore
alternative reaction structures.

Executable benchmark
- Species: X_1, X_2, X_3. Output: X_3.
- Fixed template: empty -> X_1, propensity u_1;
  X_3 -> empty, propensity u_2 * X_3.
- Evaluate all nine combinations of u_1, u_2 in {0.5, 1.0, 1.5}.
  Inputs remain constant within each scenario.
- Initial concentrations: 0.01 for every species.
- CVODE: t = 0..100, 1000 time points, rtol = atol = 1e-8.
- Target throughout each trajectory: X_3 = u_1.
- Loss: mean weighted absolute output error over all scenarios and times.
  Weights: 0.25 for the first 200 samples, 1 for the middle 600,
  and 2 for the final 200. The weighted errors are averaged, not
  divided by the sum of weights. Lower is better.
- Transient deviations count: a correct final output alone is insufficient.
- The evaluator stops trajectories reaching its all-state ceiling of 10,000.
  Low output loss does not establish bounded internal states or stability.

Design constraints
Choose exactly five distinct allowed added reactions from the supplied
order-2 mass-action library. Keep the fixed template unchanged.
Do not select null reactions or template-reserved IDs.
Supply one finite scalar rate per added reaction.
Rates are clamped to [0.001, 100]; account for this when designing.
DESIGN_CONSTRAINTS.md and the supplied library define admissibility.

Evidence-guided Decider policy
Read the current HOF overview when supplied. Further inspection of
evaluated candidates, prior LLM outcomes, diagnostics and plots is optional.
Choose promising parents by evaluated quality, not by RL or LLM origin.
Request 0 intentionally has no initial HOF: propose exploratory designs
without trying to recover withheld information.

For ten requested candidates, aim for six evidence-guided refinements
and four structurally different exploratory designs. This is guidance,
not a validity quota. Without usable parents, explore instead.
Spread refinements across promising parents where possible.

Use observed limitations to motivate rate changes, reaction replacements
or compatible mechanism combinations. Specify each resulting CRN fully.
Do not fill the batch with trivial rate variants or identical candidates.
Respect explicit topology exclusions.

The host may allocate AdaGrad rate fitting to selected structures.
Provide plausible complete rates nevertheless: fitting is limited and
neither selection for fitting nor improvement is guaranteed.
Distinguish supplied-rate results from fitted results when records allow.

For each proposal, briefly identify its parent or evidence source,
the intended change, and the uncertainty. Aggregate loss alone cannot
identify a specific scenario failure. Do not attribute improvement to one
edit when several changes were made together.

Consider feedback, integral-like, buffering or other admissible mechanisms
without assuming any mechanism class must be best. Avoid optimizing only
one disturbance condition. Inspect internal-state behavior when available,
but do not invent diagnostics or introduce an unreported scoring objective.

Submit the batch for external evaluation and use returned evidence in
later rounds. Do not request new evaluations to pre-screen proposals.
