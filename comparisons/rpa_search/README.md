# Search Comparison Benchmarks

This folder benchmarks search methods with a shared evaluator and shared budget
accounting. It currently includes:

- the input/output RPA task used in `apps/04_RPA.ipynb`
- a 4-input LogicCircuit task from `apps/01_LogicCircuits.ipynb`

The benchmark keeps the RPA IOCRN evaluator fixed:

- species: `X_1`, `X_2`, `X_3`
- input-driven production: `empty -> X_1` with `u_1`
- input-driven degradation: `X_3 -> empty` with `u_2`
- output: `X_3`
- target: output tracks `u_1`
- search space: added mass-action reactions among the three species

The LogicCircuit task uses 4 binary inputs and target formula:

```text
(u1 and u2) or (u2 and u3) or (u3 and u4)
```

For both tasks, one reported `ode_simulations` unit means one full candidate
evaluation across all task scenarios. The raw scenario workload is still saved:
`scenario_count` is the number of scenarios per candidate and
`scenario_evaluations` is the cumulative scenario count.

Each runner writes incremental progress rows so interrupted runs still leave
usable data.

## Outputs

For each run, files are written under `data/raw/<method>/<run_id>/`:

- `config.json`: resolved configuration
- `progress.csv`: one row per logging step
- `candidates.csv`: evaluated candidate summaries
- `best_network.txt`: readable best IOCRN
- `best_network.json`: reaction IDs/parameters for the best IOCRN

Plots are written to `figures/`.
Loss plots use a log-scale y-axis by default.

## Fairness Checks

Check that all methods receive the same full-simulation budget:

```bash
python3 comparisons/rpa_search/scripts/check_fairness.py --config comparisons/rpa_search/configs/rpa_100k.json
python3 comparisons/rpa_search/scripts/check_fairness.py --config comparisons/rpa_search/configs/logic_100k.json
```

The current long-run configs use 102,400 full simulations:

- random search: `candidate_budget = 102400`
- CircuiTree: `mcts_iterations = 102400`
- ReactionNetworkEvolution.jl: `candidate_budget = 102400`
- RL4CRN: `epochs = 100`, `batch_size = 1024`

## Short Smoke Tests

From the repository root:

```bash
python3 comparisons/rpa_search/scripts/run_random_search.py --config comparisons/rpa_search/configs/rpa_smoke.json
.venv/bin/python comparisons/rpa_search/scripts/run_circuitree.py --config comparisons/rpa_search/configs/rpa_smoke.json
python3 comparisons/rpa_search/scripts/run_reaction_network_evolution_jl.py --config comparisons/rpa_search/configs/rpa_smoke.json
python3 comparisons/rpa_search/scripts/run_rl4crn.py --config comparisons/rpa_search/configs/rpa_smoke.json
python3 comparisons/rpa_search/scripts/plot_results.py --config comparisons/rpa_search/configs/rpa_smoke.json
```

LogicCircuit smoke tests use `configs/logic_smoke.json` with the same runner
commands.

The random runner is a dependency-free local baseline. The CircuiTree runner
uses a custom grammar over reaction IDs and calls the same evaluator. The
RL4CRN runner uses this repository's native trainer. The Julia runner uses
ReactionNetworkEvolution.jl evolutionary operators with the same Python/Julia
task objective and budget semantics.
