# Reproduce the current paper

This directory contains the analysis entry point, plotting code and Logic reduction helpers for the current FullDuplexCRNSearch manuscript. Manuscript sources and previous-paper analysis pipelines are not needed.

## Rebuild recorded results

Obtain and extract the compact data release `fullduplex-reproducibility-20261014.tar` (about 22 GB). The public download link is pending. From the source repository root:

```bash
bash reproduce/reproduce.sh all --data-root /path/to/fullduplex-reproducibility-20261014
```

The tested offline environment is Linux x86_64 with Python 3.10 and its `venv` module. The launcher creates `reproduce/.venv` using wheels in the data release. Ensure `python3` selects Python 3.10, or create the environment explicitly first:

```bash
python3.10 -m venv reproduce/.venv
```

No model endpoint, GPU, Julia or network connection is required. `pdftoppm` from Poppler is optional and enables pixel comparisons against the reference PDFs. Individual stages:

```bash
bash reproduce/reproduce.sh verify --data-root /path/to/release
bash reproduce/reproduce.sh analyze --data-root /path/to/release
bash reproduce/reproduce.sh figures --data-root /path/to/release
```

Use `--output /path/to/results` to choose an output directory; the default is `reproduce/output/`. The entry point reads the supplied data release without modifying it. Figure and analysis workspaces combine its frozen inputs with **the analysis scripts in this checkout**. Local implementation modules come from this checkout as well. Access to the manuscript directory and original campaign locations is blocked during the analysis workflow.

The reports are `verification.json`, `scientific_checks.json`, `eil_statistics_recomputed.json` and `figure_report.json`. Figures appear under `figures/`. Nine plotting scripts generate sixteen figures; the architecture diagram is copied from the supplied reference. `manifest/figure_inputs.json` and `COHORTS.txt` in the data release document the inputs and cohort membership.

The checks reconstruct numerical and communication convergence, EIL discovery histories and statistics, Logic reduction/robustness counts, RPA structural/balance screens, parity responses, and resource accounting. They operate on recorded experiments and saved diagnostic results; they do not rerun every simulation. Full stochastic searches are a separate workflow.

## Code and data boundaries

- `analysis/biomolecular_full_duplex_rewrite/`: current plotting scripts and the shared reaction helper. The directory name preserves relative paths in the frozen inputs; it contains no manuscript.
- `diagnostics/logic_pruning.py`: replay, correctness and reduction helpers used by `comparisons/search_exploration/run_logic5_portfolio_diagnostics.py`.
- `reproduce.py`, `scientific_checks.py`: portable data checks and figure regeneration.
- `requirements-analysis.txt`: pinned analysis dependencies.

The separate data release supplies `data/`, frozen analysis inputs, manifests, reference figures, wheels, numerical runtime sources and per-run source snapshots. It preserves historical experiment provenance, including baseline data needed for comparisons in the current paper. Older manuscript builders and exploratory outputs are excluded from this source branch. The original data-release construction scripts remain in the frozen archive's `release_tools/` for auditing its selection and export procedure.

## Prepare a new matched search

For named presets, single seeds, queued campaigns, model selection through the Harness, progress monitoring and analysis of new outputs, use the **[search experiment guide](SEARCH.md)**:

```bash
python reproduce/search.py list
```

The lower-level `prepare-run` interface below remains available for individual archived Flash runs.

After setting up the analysis environment above, prepare full-duplex Logic seed 1 from its saved scientific configuration:

```bash
reproduce/.venv/bin/python reproduce/reproduce.py prepare-run \
  --data-root /path/to/release \
  --run 79f9fb37019a38f9 --output /path/to/new-experiment
```

The preparation command supports archived Flash RPA/Logic runs with complete source overlays. It deliberately restores the original sources from the release rather than substituting revised code. It launches no search and makes no API calls. Follow the root README for installing Python, Julia and the harness in the prepared directory, then run its `run.sh`.

Some retained original campaign launchers have frozen-source checks or original-machine defaults. Use `prepare-run` for a matched recorded experiment; do not assume an old campaign launcher can be run unchanged on a new machine. Matching seeds and prompts does not guarantee identical stochastic outcomes with asynchronous LLM services.

## Release verification

The current-paper branch was checked from a copy containing only staged files, with no `paper/` directory. All 17 figure PDFs matched the reference rendering, and the numerical, communication, EIL, Logic robustness, RPA structural-screen, parity and resource-accounting checks passed. The retained Python tests passed (250 passed; 24 optional-runtime tests skipped). The shell launcher was also exercised with a fresh offline analysis environment.

When testing a development working directory that still contains ignored historical files, select the tracked tests explicitly after installing the numerical dependencies and pytest:

```bash
python -m pytest $(git ls-files 'tests/test_*.py' 'literature_rag/tests/test_*.py')
```

Original manuscripts and excluded scripts can remain locally; their presence is not required by the release. The compact archive is unchanged, including the frozen sources used to prepare matched reruns.
