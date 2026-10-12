# RPA exploration run without LLMs

## Purpose

Run the new structure-search and bounded rate-fitting pipeline for a substantial
budget after the NN-size pilot. This is one exploratory seed, not a controlled
comparison against legacy RL or evidence that a 1024-dimensional network is
equivalent to the larger network. No manuscript results are changed.

## Frozen Settings

- Seed 0, 100 epochs, 1023 proposals per epoch, intermediate NN dimension 1024.
- Evolutionary log-rate fitting, at most 16 objective calls per proposal.
- At most 1,636,800 physical CVODE evaluations; cache hits do not consume this
  budget. Report actual evaluations, not only proposal counts or epochs.
- Rate bounds [0.001, 100], initial rates 1; fit/evaluation deadlines 60/30 s.
- Source probabilities: RL 50%, archive mutation 25%, random 25%. Actual source
  counts fluctuate; archive mutation falls back to random when no parent exists.
- Discovery archive 2000, elite imitation top 30 with up to 16 demonstrations.
- RPA qualification threshold 0.016491168958935352, unchanged from the pilot.
- 48 CPU evaluator workers, one numerical library thread per process.
- Harness disabled: no LLM, API calls, proposal banks or RAG. This is a mixed
  RL/mutation/random optimizer, not the legacy RL-only control.
- Comet project `genai-net-exploration`; configuration, metrics and plots only.
  Automatic code, Git and host metadata logging remains disabled.

## Artifacts And Launch

Campaign root: `/local1/rossin/search-exploration/rpa-no-llm-100epoch-20260914`.
Run output: `run/`; frozen settings: `exploration.json` and `task.json`;
source archive: `source.tar.gz`; combined output: `run.log`.
The runner records source hashes, evaluator identity and full effective settings
in `run/manifest.json`. The original scientific sources remain unchanged.

From `/local0/rossin/git/GenAI-Net`, launch in tmux session
`rpa-no-llm-100epoch-20260914` with:

```bash
env -u COMET_API_KEY CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/local0/tmp/mpl \
  .venv/bin/python -m comparisons.search_exploration.run \
  --task rpa \
  --task-config /local1/rossin/search-exploration/rpa-no-llm-100epoch-20260914/task.json \
  --exploration-config /local1/rossin/search-exploration/rpa-no-llm-100epoch-20260914/exploration.json \
  --embedding-size 1024 --device cpu \
  --comet-project genai-net-exploration \
  --output /local1/rossin/search-exploration/rpa-no-llm-100epoch-20260914/run
```

The environment override selects the already-configured Comet credential source;
no credentials are embedded in the command. There is no `--harness` flag.
The run output must not exist at a fresh launch. No automatic retries are used.
Explicit `--resume` requires a compatible checkpoint and unchanged manifest;
failed cohorts require separate diagnosis and authorization before retry.

## Monitoring And Interpretation

Inspect `run/status.json`, `run/metrics.jsonl`, `run/comet.json` and `run.log`.
Figures `run/progress.png`, `run/progress.pdf`, and `run/sources.png` are refreshed
every five epochs. `run/results.sqlite` is the compatibility database.
Full cache and budget accounting
are in `run/exploration.sqlite`.

Primary outcomes are canonical loss and qualifying archived topology count,
tracked against actual evaluation cost. Also inspect source contributions,
descriptor coverage, cache reuse and fitting failures. Descriptor coverage is
not a formal measure of mechanistic diversity. A larger batch changes the
learning schedule relative to the small pilot, so this is not a size-only test.

The completed RPA pilot used roughly 2.7 GiB per 10,200 evaluations. Linear
extrapolation suggests around 430 GiB for this run, but this is not a guarantee;
trajectory compressibility varies. Outputs use `/local1`, which had about 1.5 TiB
free at preflight. `/local0` has limited free space and is not used for results.
The GPU driver was unavailable at preflight; no driver changes were attempted.
