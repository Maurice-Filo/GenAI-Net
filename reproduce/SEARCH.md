# Repeat the search experiments

`search.py` prepares and runs **new searches**, then analyses their outputs. To rebuild figures from recorded experiments instead, use [the analysis workflow](README.md).

The launcher restores each selected seed's archived source, task configuration, search hyperparameters, fitting recipe and exact prompt. It verifies their hashes before execution. The numerical algorithms are unchanged. Model routing is supplied by the user; the Harness owns the model's credentials, endpoint, context limits and reasoning/sampling settings.

## Install once

From the repository root, follow the [Python and Harness installation instructions](../README.md#install-the-source-code), then install the small additional dependencies:

```bash
source .venv/bin/activate
python -m pip install -r reproduce/requirements-search.txt
source "$HOME/.local/share/fullduplex-crn/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Extract the compact data release and set its path:

```bash
export FULLDUPLEX_DATA_ROOT=/path/to/fullduplex-reproducibility-20261014
python reproduce/search.py list
```

The data release supplies the exact reference configurations and source snapshots. No original `/local0/...` or `/local1/...` campaign directories are required. Prepared campaigns contain their own copies; the data release is read-only and is not needed by `run` afterward. Keep the campaign at its prepared location because generated commands contain absolute paths.

## Configure the model in the Harness

Use the Harness GUI to select or add the provider, endpoint, API key and model. Configure the **headless profile** used for search. Pass its provider ID and model ID as one name:

```bash
--model deepseek-official/deepseek-v4-flash
--model my-local-provider/my-model-id
```

The text before the first slash is the provider ID; everything afterward is the model ID (which may itself contain slashes). These are identifiers, not display names. A bare model ID is also accepted when it can be resolved unambiguously from the saved Harness configuration. Qualified names are preferable when using multiple providers.

The CLI does **not** configure or overwrite provider settings, read your credentials file, or substitute an endpoint. In particular, it disables the old runners' automatic localhost-provider configuration. Changing the model does not automatically change task prompts or numerical settings. `rpa-local` and `logic-local` select the reference campaign settings, not a provider: you still supply `--model`.

## One experiment

```bash
python reproduce/search.py prepare \
  --experiment logic-full --seed 1 \
  --data-root "$FULLDUPLEX_DATA_ROOT" \
  --model deepseek-official/deepseek-v4-flash \
  --output "$HOME/fullduplex-runs/logic-full"
```

`DSH_HOME` comes from the Harness installer environment. Alternatively supply `--dsh-home /path/to/harness-home`. Preparation creates files only; it makes no model calls.

With the reference Julia version (1.12.6) on `PATH`, instantiate the prepared numerical environment. One instantiation normally supplies the cached packages needed by the other seeds' identical environments:

```bash
julia --project="$HOME/fullduplex-runs/logic-full/jobs/seed-01/source/numerical_runtime/SynthEvo" \
  -e 'using Pkg; Pkg.instantiate()'

python reproduce/search.py check "$HOME/fullduplex-runs/logic-full"
python reproduce/search.py run "$HOME/fullduplex-runs/logic-full"
```

`check` verifies source/configuration hashes, Python imports, actual Julia worker startup and the Harness installation. It makes no API requests and does not test credentials, server availability or whether a configured model is serving. `check --files-only` works without numerical dependencies. If using another Python environment, pass `--python /path/to/venv/bin/python` to `check` and `run`.

**`run` starts computation and, for LLM experiments, model requests.** It shows campaign progress in a terminal, or periodic plain-text updates when redirected to a log. Per-seed runner output is retained in `jobs/seed-NN/run.log`.

## Twenty seeds

```bash
python reproduce/search.py prepare \
  --experiment logic-full --seeds 1-20 --parallel-runs 8 \
  --data-root "$FULLDUPLEX_DATA_ROOT" \
  --model deepseek-official/deepseek-v4-flash \
  --output "$HOME/fullduplex-runs/logic-full-20"

python reproduce/search.py run "$HOME/fullduplex-runs/logic-full-20"
python reproduce/search.py status "$HOME/fullduplex-runs/logic-full-20"
```

`--parallel-runs` controls concurrent seeds. Each seed also has its recorded numerical `search_workers` and `optimization_workers`; these are additional processes. Start with one concurrent seed on a smaller machine. The Harness has its own model-request concurrency controls.

The controller runs in the foreground; use your normal terminal multiplexer or batch scheduler for unattended operation. It continues remaining seeds after an individual failure, reports failures at the end, and preserves every attempt. Re-running the command skips completed/failed/interrupted attempts and launches only untouched pending seeds. It does not silently restart a failed search from epoch zero. Prepare failed seeds in a new campaign directory if you want new attempts. Ctrl-C terminates the active child process groups and records interruption.

## Available experiments

All presets use the recorded seed-specific settings, normally 100 epochs. Seeds can be a single number, a comma list or a range within 1–20.

| Preset | Experiment |
|---|---|
| `rpa-full`, `logic-full` | Full-duplex benchmark searches |
| `hill-full`, `ultrasensitive-full`, `biphasic-full` | Input–output response tasks |
| `classification-full`, `parity-full` | Classification and four-input parity |
| `rpa-insertion`, `logic-insertion` | LLM → numerical-search communication only |
| `rpa-feedback`, `logic-feedback` | Numerical-search → LLM communication only |
| `logic-independent` | Both streams execute without communication; **LLM requests still occur** |
| `logic-eil-zero` | Full duplex with imitation weight zero |
| `rpa-prompt-neutral` | Full duplex with the mechanism-neutral task prompt |
| `rpa-local`, `logic-local` | Original local-model reference settings |
| `rpa-local-60min` | RPA local-model follow-up with a 60-minute role deadline |
| `rpa-numerical` | Numerical hybrid without any LLM requests |
| `rpa-ga` | Standalone GA without any LLM requests or Julia requirement |

The RPA independent union was reconstructed, so there is no invented `rpa-independent` execution preset. Older reference-policy runs remain available as recorded comparison data, not as supported fresh-search presets.

## A small numerical smoke test

This tests the GA search path without requiring an LLM or Julia:

```bash
python reproduce/search.py prepare \
  --experiment rpa-ga --seed 1 \
  --data-root "$FULLDUPLEX_DATA_ROOT" \
  --batch-size 4 --evaluation-budget 8 --search-workers 1 \
  --output "$HOME/fullduplex-runs/smoke"
python reproduce/search.py run "$HOME/fullduplex-runs/smoke"
python reproduce/search.py analyze "$HOME/fullduplex-runs/smoke"
```

These are deliberately shortened test settings, not a replication of the reported search budget. No qualifying design is expected from such a small test. Standalone GA progress is measured in batches, not RL epochs.

Other explicit overrides are `--epochs`, `--batch-size`, `--evaluation-budget`, `--search-workers`, `--optimization-workers`, `--llm-every`, `--llm-candidates` and `--llm-timeout`. Unsupported overrides are rejected. The reference and effective values are recorded. Task objectives, prompts and imitation weights come from the selected preset. To change them, add a separate experiment definition rather than silently modifying a matched run.

## Storage policy

Immediately after a run finishes and its processes close, the controller applies the existing validated time-downsampling routine at stride 10. It retains losses, reactions, rates, provenance and all scalar records. Qualifying and historically elite topology traces remain protected. Compaction errors are reported independently of search success and do not stop the other seeds.

Allow at least 15 GiB of temporary free space for compaction, in addition to active-run storage. `--no-compact` at preparation disables it explicitly. Compaction logs and validation records are retained; an unresolved previous compaction marker requires inspection before another attempt.

## Analyse new outputs

```bash
python reproduce/search.py analyze "$HOME/fullduplex-runs/logic-full-20"
```

This produces `analysis/overview.pdf` and PNG, per-seed `endpoints.csv`, epoch `curves.csv`, and `topology_provenance.json`. The plots show median and interquartile range across **completed seeds only**. The report identifies excluded unfinished seeds. No old results are substituted.

The portfolio includes numerical discoveries and valid direct LLM proposals, deduplicated by sorted added-reaction IDs. Curves include numerical loss and combined-system loss separately in CSV. Terminal markers include completion of pending work and can have unequal simulation costs. Missing epoch attribution is reported and those records appear only in terminal counts. Observed provenance routes may overlap; they are not exclusive causal credits. GA x coordinates are batches.

Optional scientific diagnostics:

```bash
# Algebraic/motif screening of all qualifying RPA rate records; no new simulations.
python reproduce/search.py analyze "$HOME/fullduplex-runs/rpa-full" --rpa-screen

# Additional simulations: Logic reduction and the paper's perturbation protocol.
python reproduce/search.py diagnose "$HOME/fullduplex-runs/logic-full-20" --workers 2
```

The RPA screen uses the paper's motif and restricted integral-balance functions, counting any tested qualifying parameterization as a witness for a topology. Logic diagnostics select the best qualifying parameterization per topology, reuse the existing canonical-order recovery, subset reduction and 32-draw perturbation functions, and report unresolved baselines. These simulations can be substantially more expensive than the summary plots. Diagnostic outputs are separate from search results; they are not automatically launched at search completion.

## Provenance and reproducibility

Each seed has `reference_manifest.json`, effective task/search/recipe files, source hashes, a hash-checked `job.json`, and a restored `source/` tree. Campaign preparation refuses an existing destination. Source/input changes are rejected by preflight and worker startup.

The transport adapter changes routing arguments in the worker process, leaving archived scientific source files intact. Before a new run's identity is computed, its manifest is labelled with the **actual model/provider** and the adapter hash; the old routing metadata is retained separately as `reference_transport`. `effective_execution.json` and `campaign.json` also record the selected model and overrides. They do not store API credentials. Harness settings remain user-owned; retain a private copy of the relevant model configuration to document a reproduction, without publishing credentials.

The same seeds and scientific settings do not guarantee identical discoveries under asynchronous scheduling and changing model services. Resource allocation and explicit overrides are recorded. The launcher does not claim hardware-identical or bitwise-identical reruns.

## Validation

All 19 presets were prepared from seed 1 and passed source/input checks and the archived baseline verifier. Brief executions covered an eight-evaluation GA search, a one-epoch numerical hybrid search, Julia worker startup and a three-evaluation AdaGrad fit. Completion-time compaction and fresh-result plotting were exercised. CLI tests cover model routing/forking, metadata identities, input tampering, failed-seed continuation and topology deduplication. The numerical smoke tests do not validate scientific performance, and no live model request or full campaign was launched as part of this check.
