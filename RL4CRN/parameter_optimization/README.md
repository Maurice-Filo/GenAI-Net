# Persistent Julia Rate Optimization

Opt-in test-stage module. Existing RL/GA/LLM paths, canonical evaluators, and
campaign settings are unchanged. AdaGrad is the default; Adam is optional.

## Usage

The existing SynthEvo Julia package environment must be installed. Point to its
package directory, which contains `Project.toml`, `Manifest.toml`, and `src/`:

```bash
export SYNTH_EVO_PROJECT=/local0/rossin/git/SynthEvo/SynthEvo
```

```python
import numpy as np
from RL4CRN.parameter_optimization import (
    CRNSpec, Reaction, Experiment, JuliaOptimizer, ReactionLibrary,
)

crn = CRNSpec(("X",), (), (Reaction((1,), (0,)),))
t = np.linspace(0, 3, 61)
experiment = Experiment(
    times=t, initial_states=[[1.]], inputs=[[]], output_matrix=[[1.]],
    targets=np.exp(-.7 * t)[None, None, :], loss="l2",
)
with JuliaOptimizer() as worker:
    model = worker.prepare(crn, experiment)
    pending = worker.submit(model, [[.2], [1.5]], steps=32)
    # Other work can proceed here. This worker processes queued jobs serially.
    result = pending.result()
    first = result["results"][0]
    continued = worker.optimize(model, states=[first["state"]], steps=32)
```

For an existing starter network and indexed library:

```python
library = ReactionLibrary.from_iocrn(starter_crn, reaction_library)
crn = library.select(reaction_ids)  # rates follow exactly this ID order
model = worker.prepare(crn, experiment)
```

The adapter precomputes species/reactant/product/input mappings once. Selection
does not clone the IOCRN or compile symbolic equations. It enforces distinct IDs,
rejects the empty reaction, and rejects added reactions with the same reactant
and product counts as a starter reaction, regardless of input modulation.
The general `CRNSpec` separately permits arbitrary fixed/free reaction mixtures
and retains input modulation in reaction identity. It is not itself a benchmark
candidate validator. Non-mass-action kinetics are rejected by the adapter.

## Mathematical Contract

For reaction j with net change nu_j, multiplicities a_ij and optional input u_c:

    v_j(x,k,u) = k_j * product_i(x_i ** a_ij) * u_c
    dx/dt = sum_j nu_j * v_j
    dS/dt = (df/dx) S + df/dk,    S(t_start) = 0

The input factor is 1 for an unmodulated reaction. Repeated species use plain
powers, without factorial normalization, matching `MassAction.propensity`.
Only free parameters receive sensitivity columns. Fixed rates are untouched.
Inputs and initial conditions are independent of fitted rates in this version.

Experiments contain explicit paired scenarios, not an implicit Cartesian product.
`initial_states` correspond to `times[0]`. All inputs are constant over that
scenario. Outputs are a general linear map Cx. Targets can vary over time.

    L = mean_s,o,t [ w_s,o,t * |(Cx - target)/scale|^q ]

q is 1 or 2. `scale=1`, or `max(abs(target), 1e-6)` for relative error. The
denominator is the number of scalar observations, NOT the sum of weights. This
matches the existing weighted tracking loss. L1 uses sign(0)=0 as its zero-error
subgradient; there is no smoothing, surrogate reward, pruning, or regularizer.

### No Sparsity Objective During Fitting

Task L1 means absolute output-tracking error, not an L1 penalty on the rates.
The fitting objective is the task loss alone: no term such as lambda*sum(abs(k)),
no rate-threshold pruning, and no deletion of reactions. Importing SynthEvo's
AdaGrad/Adam update helpers does not import its regularization loss or pruning
heuristics. An uncoupled/unobserved reaction has zero task gradient and retains
its rate, even at the lower bound. Both optimizer step policies test this.

Reaction simplification is implemented as a separate, opt-in postprocessing
pilot in `search_streams/pruning.py`. Preserve the original CRN and loss; explicitly remove candidate
non-template reactions, optionally refit surviving rates within the same bounds,
and reevaluate against the canonical task with a predefined performance tolerance.
Removing a reaction changes topology; setting its rate to zero would violate the
current positive-rate contract. Record the simplified network as a derived design
with its own evaluation cost, without rewriting search results or silently changing
the training objective. Low rate magnitude alone is not proof of dispensability.
It is never enabled inside the rate optimizer; see the Search Pilot section below.
Finite differences near good RPA solutions can cross L1 kinks; use convergence
checks on the differencing step and independent smooth-region checks.

Optimization is in log coordinates theta=log(k), with gradient k*dL/dk. Each
update projects theta onto [log(.001), log(100)] by default. Positive out-of-bound
starts are clipped and flagged. Zero, negative, nonfinite, or missing start values
produce `invalid_start` individually. Set bounds explicitly for another domain.
AdaGrad learning rate defaults to 0.1 and epsilon to 1e-8. Adam uses beta1=0.9,
beta2=0.999 and epsilon=1e-8. No automatic restarts are generated.

## Execution And Reuse

- One Julia process per `JuliaOptimizer`, one Julia/BLAS thread. Use separate
  instances for concurrent work; this low-level interface creates no CPU-wide pool.
- JSON-lines IPC carries complete jobs, not every optimizer step. `submit`
  returns a Future; it does not imply concurrent starts inside that worker.
- Analytic numerical mass-action sensitivities implement SynthEvo's forward
  equations. The optimizer includes the existing `src/CRNOptim.jl` helpers
  directly. It does NOT call `symbolic_gradient_descent`, its IFF-specific loss,
  its shared-prehistory protocol, or its pruning machinery.
- Compiled kernel types are independent of topology size; reaction metadata and
  ODE problems are cached. Preparing a new topology does not invoke MTK/Catalyst
  symbolic compilation. CVODE solver workspaces are still initialized per solve.
- Default solver is Julia Sundials CVODE_BDF, rtol=atol=1e-8. Tsit5 is opt-in.
  This is an augmented sensitivity ODE, so adaptive steps need not match the
  Python state-only CVODE solver bit for bit. Canonical reevaluation remains
  required before inserting optimized candidates into a search campaign.
- Set `max_species` to the task's instability ceiling. RPA uses 10,000, with a
  continuous terminal event on any species, not just observed outputs. The
  general interface defaults to no upper ceiling. It must be supplied explicitly
  for bounded benchmark tasks. Python CVODE also uses nonnegativity inequality
  constraints; this prototype checks Julia's saved species states instead, so
  it is not a bit-identical replacement for the canonical evaluator.
- The process loads installed dependencies by UUID from SynthEvo's existing
  Manifest, following its lightweight compiled-test approach. It neither installs
  packages nor changes that environment. Julia may write its usual user cache.
  A standalone pinned Julia environment is a later packaging task.
- Prepared handles hash the complete ordered CRN and experiment. The default
  cache holds 32 models, evicts least-recently-used handles, and rejects stale
  handles. This bounds model count, not bytes. Re-prepare an evicted model.
- Solver states/sensitivities stay Julia-side unless trajectories are explicitly
  requested. Optimization returns scalar histories, counters, and checkpoints.

## Results, Failure And Resume

Each start reports status, initial clipping, best evaluated rates/loss, separately
evaluated final rates/loss, scalar history, and optimizer state. State includes
AdaGrad accumulators or Adam moments, step count, last loss/gradient, and best-so-
far values. Continuing N+M steps matches an uninterrupted N+M-step job; the saved
last gradient avoids a duplicate solve. Checkpoints can be JSON-serialized and
restored in an identical backend with the same model/configuration. Checksums
detect accidental modification; they are not a security authentication scheme.

`evaluations` counts attempted joint loss/gradient evaluations, including failed
ones; `completed_evaluations` counts successful ones. `scenario_solves` counts
actual attempted scenario solves. Cached continuation boundaries are not charged
again. These augmented solves cost more than state-only evaluations: raw counts
alone are NOT a fair compute-budget comparison with GA/Powell/RL.

Unsuccessful solver return codes, incomplete trajectories, nonfinite states or
gradients, and sampled species below -10*atol produce `evaluation_failed` for that
start. Signed sensitivities are not subjected to the species positivity check.
Species are never clipped. This tests saved states, not the entire interpolant.
Unclassified exceptions propagate as job errors rather than concealing bugs.
A failed final iterate is never mislabeled as successfully evaluated; a previous
best may still be present with the failure status.
Reaching `max_species` is also an explicit failure. By default the optimizer stops
that restart; it does not backtrack, relax the ceiling, or retry silently. A large
AdaGrad step can encounter this boundary despite a small starting output loss.

### Opt-In Feasibility Backtracking

```python
result = worker.optimize(
    model, starts, steps=32, learning_rate=.1,
    step_policy="feasible", max_backtracks=8, max_evaluations=65,
)
```

This separate implementation leaves the original `step_policy="stop"` loop
unchanged. It tests theta' = project(theta - 2^(-j) * delta), j=0..max_backtracks,
using one tentative AdaGrad/Adam update. Rejected trials do not repeatedly update
the accumulator/moments. The first feasible trial commits both rates and optimizer
memory. There is no sufficient-decrease criterion: feasible loss increases are
allowed, with the best evaluated point retained separately.

Each trial records rates, scale, status, error, loss if valid, attempted joint
evaluations, scenario solves, and elapsed time. Known EvaluationFailure outcomes
are recoverable; unexpected exceptions propagate. Initially infeasible starts
still fail individually without invented gradients or automatic reseeding.
`backtracking_exhausted`, `evaluation_budget_exhausted`, and `stalled` preserve
the last accepted state and its valid final loss, but do not count as completed
requests. No rejected point enters the returned best. The evaluation cap is per
start per call, including failed calls, and counts a fresh initial evaluation;
continuations reuse the saved current gradient. Passing None removes this cap,
but finite retries, requested steps, and the hard wall deadline still apply.

The added source is separately fingerprinted in checkpoints. Opt-in continuation
also requires unchanged backtracking settings. Changing this implementation
invalidates older checkpoint fingerprints explicitly rather than silently mixing
backends. Ordinary calling conventions and result keys remain compatible.

`timeout` is a hard wall deadline per dispatched job, including its first JIT and
all starts. Queue/lock waiting is outside that deadline. Exceeding it stops the
worker and raises `TimeoutError`; no fabricated score or automatic retry is used.
An interrupted job has no partial result stream yet. Recreate the worker and
resume only a previously returned checkpoint. Context-manager exit stops/reaps
the process and cancels queued work.

## Verification

```bash
.venv/bin/python -m pytest tests/test_julia_parameter_optimization.py -q
env SYNTH_EVO_PROJECT=/local0/rossin/git/SynthEvo/SynthEvo \
  .venv/bin/python -m comparisons.search_exploration.benchmark_julia_rates \
  --output comparisons/search_exploration/reports/julia-rate-prototype-UNIQUE \
  --comet-project genai-net-exploration
```

Real Julia tests run only when `SYNTH_EVO_PROJECT` is set; otherwise they skip
explicitly. Tests cover analytic dynamics/gradients, canonical Python CVODE,
repeated reactants, zero states, fixed/input-modulated reactions, permutations,
relative/multiple-output losses, L1/L2, bounds, bad starts, solver failure,
timeouts, exact continuation, independent worker restore, and cache eviction.
The microbenchmark records versions/source hashes, cold/warm timings, complete
model/experiment definitions, optimizer traces, and canonical best reevaluations.
It is software verification on one selected topology, not evidence that AdaGrad,
Adam, or this wrapper beats the previous search methods.
The Comet flag is optional and uses existing credentials only; metrics, status,
the figure, and the short report are logged, with automatic environment/code
capture disabled. Local raw optimizer results are saved before canonical checks.
A known guard-stopped restart is reported as such even when its retained best
passes canonical verification; that does not mean all requested updates finished.

Not yet integrated: RL/GA scheduling, task-wide screening, SSA, piecewise inputs,
arbitrary nonlinear task objectives, and process-pool throughput benchmarks.
## Search Pilot

The opt-in asynchronous integration and separate explicit reaction-deletion
postprocessing are documented in [../search_streams/JULIA_PILOT.md](../search_streams/JULIA_PILOT.md).
The rate optimizer itself still has no sparsity/pruning penalty.
