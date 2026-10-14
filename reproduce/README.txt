FullDuplexCRNSearch: compact reproducibility release
==================================================

Start here
----------
This is a local scientific-data release, not the manuscript or a new search run.
The repository's reproduce/ folder contains the packaging templates; run the
commands below from the prepared release directory, where data/ is present.
Use Linux x86_64 with Python 3.10, the Python venv module, and sufficient space
for the unpacked data plus approximately 1 GB for the analysis environment.
Poppler's pdftoppm is optional; it enables rendered-PDF comparison.

From this directory, one command installs the cached analysis dependencies,
checks the exported records, recomputes the principal scientific analyses,
and rebuilds the manuscript figures:

    bash reproduce.sh all

No model credentials, LLM endpoint, GPU, Julia installation, or internet access
are needed for this workflow. It can take several minutes. The full integrity
check reads every retained table and uses eight worker processes.

Individual stages:

    bash reproduce.sh verify
    bash reproduce.sh analyze
    bash reproduce.sh figures

Outputs are in output/. Reports are verification.json, scientific_checks.json,
figure_report.json, and eil_statistics_recomputed.json. Rebuilt PDFs are under
output/figures/. Source records and frozen reference inputs are never overwritten.
The temporary output/work/ tree can be removed after inspecting the figures.

For other platforms/Python versions, create an environment and install with
pip install -r requirements-analysis.txt (internet may be needed), then run
python reproduce.py all. Cached wheels target the tested Linux/Python platform.

Contents and counting units
--------------------------
manifest/runs.json identifies 521 distinct source runs, with campaign membership
and original location aliases. These are not 521 interchangeable replicates:
each analysis uses its own explicit cohort. In particular, the LLM accounting
cohorts comprise 17 groups of 20 runs (340 runs); the remaining records include
numerical and GenAI-Net references. Follow the frozen selections in analysis/.

data/runs/<id>/ contains each run's configuration, epoch history, completion
records, direct proposals and other available compact metadata. For 401 runs
with exploration databases it also contains lossless gzip JSONL exports of
designs, evaluation_summaries, events, meta, processing and visits. The other
120 runs are legacy references represented by their available saved records.

All rows of the exported tables are retained, including non-qualifying designs
and designs no longer in the final archive. Reaction IDs, rates, losses,
qualification flags and recorded lineage remain available. export.json records
each table's schema, columns, row count and SHA-256 of the uncompressed lines.
Each line is a JSON array in the recorded column order; embedded payload JSON
is preserved verbatim. Original SQL blobs, if any, use {"base64": "..."}.

Large trajectory/evaluation blobs, model/policy checkpoints, temporary Julia
jobs and duplicated harness workspaces are omitted. This is not a checkpoint
resume archive. Full cached dynamics for every design are not included.
The stored reaction/rate records support resimulation with the task evaluator.

data/diagnostics/logic/ retains the frozen selection and per-candidate reduction
and perturbation results. data/diagnostics/rpa/ retains the qualifying portfolio
records, reaction catalog and structural-screen evidence. data/campaigns/ holds
available campaign plans, matched configurations and task prompts. data/prompts/
contains the manuscript's prompt text.

data/llm_usage/session_events.jsonl.gz contains token-usage records and tool-call
identifiers/names for 6,706 recorded sessions. Conversations, tool arguments,
tool contents and credentials are excluded. Original source log hashes are
retained. The package recomputes 59,539 usage events and 104,008 tool calls.
Resource coverage follows the saved accounting cohort: absent log usage is not
invented. Cost values are published price-schedule estimates, not invoices;
local model hardware/electricity costs are not assigned API prices. See
analysis/biomolecular_full_duplex_rewrite/token_costs/README.txt for definitions.

analysis/ contains frozen plot inputs and the plotting/analysis programs.
expected/figures/ contains the 17 manuscript figures used for visual checking.
One is a supplied architecture diagram; the other 16 are rebuilt from data.
manifest/figure_inputs.json maps figure filenames to scripts and frozen inputs;
COHORTS.txt and manifest/cohorts.csv explain the overlapping run selections.
RL4CRN/, comparisons/ and tools/ contain the scientific implementation and
supporting scripts. The manuscript text is deliberately not bundled.

What the checks establish
-------------------------
verify independently decompresses every exported table, verifies line hashes
and row counts, and recounts qualifying topology identities and best losses.
When present, manifest/checksums.json covers the distributed payload files.

analyze reconstructs all 160 RPA/Logic communication trajectories from epoch
and direct-proposal records; recounts the EIL ablation's 40 portfolios and their
epoch histories; recomputes EIL statistics; checks canonical Logic reduction
and robustness counts from the diagnostic records; recomputes RPA structural
and algebraic screens for 100 communication and 40 prompt-ablation portfolios;
and reaggregates token usage and tool calls from compact event records.

It also checks numerical/breadth convergence against the raw progress histories
and validates Parity topology qualification and saved truth-table responses.

figures uses frozen plot inputs and verifies terminal endpoints against released
run records. It blocks reads from the original source campaign directories.
When pdftoppm is installed, figure_report.json reports pixel equality at 100 dpi
against the supplied PDFs. Differences in PDF metadata are irrelevant.

These checks reproduce analysis of the recorded experiments. They do not rerun
hundreds of stochastic searches, refit all designs, or repeat every perturbation
simulation. Independent generation of a new cohort is a separate workflow.

Using existing SQLite-based analysis code
----------------------------------------
To materialize one retained database schema from its portable export:

    python reproduce.py materialize --run RUN_ID --output /tmp/run.sqlite

This refuses to overwrite an existing database. Omitted trajectory/cache tables
are not fabricated; code needing them must resimulate or use released diagnostic
records. The exported schema and line records are the authoritative compact copy.

Running new searches
--------------------
The source tree, per-run manifest.json, frozen source.tar.gz where available,
campaign configs and prompts preserve the recorded settings. The analysis
environment is intentionally smaller than the numerical/LLM runtime.
Follow tools/HARNESS_SETUP.md and tools/install_harness.sh for harness setup.
The original measured Python environment is recorded separately in
manifest/python-environment.txt; the production Julia project and its lockfile
are retained under numerical_runtime/SynthEvo/ (Julia 1.12.6). Endpoint paths and output destinations must
be configured for the new machine. Never reuse the original absolute output
paths embedded in old campaign plans.

The Julia project snapshot is captured at release preparation time; historical
run manifests identify its project path but do not individually hash that external
checkout. The recorded-result analysis workflow does not depend on rerunning it.

Prefer the campaign-specific source/ snapshot and per-run source.tar.gz when
reconstructing an experiment. The live repository has evolved since the older
campaigns; its legacy freeze check detects a later breadth-task change. The
saved EIL and prompt-ablation source snapshots pass their original freeze checks.

For a Flash RPA/Logic run with an archived source overlay, prepare a fresh working
directory with the exact saved scientific source/configuration and a launcher:

    python reproduce.py prepare-run --run RUN_ID --output /path/to/new-experiment

This command restores the common frozen baseline and overlays the run's saved
sources, then checks their hashes. It does not launch a search. After installing
the numerical dependencies and configuring your LLM harness, execute from the
new directory:

    DSH_HOME=/path/to/configured/harness bash run.sh

The launcher uses the selected run's seed, prompts, worker counts and search
hyperparameters. Its destination new_run must not already exist. New API calls
may incur costs. The model deployment is determined by your harness configuration;
check it against replay_config/reference_manifest.json before running.

Re-running a search requires numerical dependencies and, for communicating
arms, an LLM deployment/API account. Asynchronous scheduling and model service
behavior mean that matching seeds does not guarantee bitwise-identical searches.
No automatic paid campaign is launched by reproduce.sh.

Release status
--------------
This directory is prepared for local verification and eventual publication.
Original paths and author/source metadata are retained for traceability: this
is not an anonymous submission artifact. No data have been uploaded. Use the
validation reports and manifest/payload_summary.json to inspect the final size
and coverage before depositing the release.
