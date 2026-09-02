# GenAI-Net Literature Pilot

This folder contains a deliberately small, reproducible literature retrieval
system for the design tasks evaluated in the GenAI-Net paper. It uses only:

- Europe PMC open-access JATS full text;
- a local SQLite database with FTS5 search; and
- one read-only command that an agent can call on demand.

No API key, model call, vector database, background server, or package install is
required. Downloaded articles and the generated database are ignored by Git.

## Corpus scope

`topics.json` targets 500 articles across synthetic biology, general CRN design, logic circuits, oscillators,
deterministic robust perfect adaptation, stochastic RPA/noise control, dose
response, classification, tracking, and habituation. A paper can be associated
with more than one task, while each PMCID is stored only once. The searches
require the domain and task concepts to occur in the title or abstract. Nominal
quotas guide the balanced first pass. The broad synthetic-biology pool supplies
replacements when a narrow task category has fewer open-access records, avoiding
overfilling ambiguous categories such as generic noise or animal habituation.

Only records satisfying both `OPEN_ACCESS:Y` and `IN_PMC:Y` are downloaded. The
manifest records the source URL and SHA-256 checksum for every article.

## Build the corpus

From the repository root:

```bash
python3.11 literature_rag/fetch_papers.py --limit 500
```

The command is resumable. Re-running it skips downloaded PMCIDs and repairs
topic associations. Files are stored under `papers/`; metadata and full-text
search live in `index/literature.sqlite3`.

## Search

Human-readable output:

```bash
python3.11 literature_rag/search.py \
  "antithetic integral feedback robust perfect adaptation"
```

Machine-readable Harness output:

```bash
python3.11 literature_rag/search.py \
  "synthetic genetic logic gate" --topic logic_circuits --limit 8 --json
```

The Harness should be granted read-only access to this folder and instructed to
treat returned passages as literature evidence, not as simulation results. It
must cite the returned DOI or PMCID and may not modify `papers/` or `index/`.

### Sandboxed Harness runs

If the Harness cannot execute shell commands, start the read-only queue sidecar
against a queue inside its run workspace:

```bash
python3.11 literature_rag/serve.py --queue /path/to/run/LITERATURE_REQUESTS
```

The agent writes `search-01.request.json` containing:

```json
{"query": "antithetic integral feedback", "topic": "robust_perfect_adaptation", "limit": 8}
```

It then reads `search-01.response.json`. Stop the sidecar with `Ctrl-C`. The
sidecar opens the literature database in SQLite read-only mode and has no model,
API key, or network dependency.

## Inspect the corpus

```bash
sqlite3 literature_rag/index/literature.sqlite3 \
  "SELECT COUNT(*) AS papers FROM papers;"

sqlite3 literature_rag/index/literature.sqlite3 \
  "SELECT topic, COUNT(*) FROM paper_topics GROUP BY topic ORDER BY topic;"
```

This lexical retrieval baseline is intentionally simple. PaperQA2 or semantic
embeddings can be layered on the same `papers/text/` corpus later if benchmark
results show that lexical retrieval is insufficient.
