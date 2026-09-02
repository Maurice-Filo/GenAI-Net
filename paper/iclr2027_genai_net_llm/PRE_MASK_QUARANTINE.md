# Pre-template-mask quarantine

Date: 1 September 2026

## Status

The quarantine completed successfully. No run or result artifact was deleted.
All large objects were moved with same-filesystem renames, so their bytes were
not copied or modified.

- Quarantine root: `/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/quarantine/2026-09-01-pre-template-mask`
- Authoritative manifest: `manifest.json` under that root
- In-repository inventory: `generated/pre_mask_quarantine_inventory.json`
- Recorded moves: 551
- Recorded files: 572,808
- Recorded size: 73.37 GiB
- Verification: every destination exists, every original source is absent, and
  every manifest entry is marked applied.

The manifest records the original path, destination, artifact kind, byte and
file counts, reason, inferred task where available, and applied state for each
move. It is the restoration map if an artifact is needed for forensic analysis.
Quarantined artifacts must never be added to a paper cohort or used to regenerate
submission figures.

After the manifest closed, a small forensic copy of the pre-v2 `main.tex` and
`sections/` was added under
`derived-results/manuscript-source-pre-v2/`. It is intentionally outside the
active paper and outside the 551-move count; it preserves the withdrawn prose
while the live TeX is reduced to method text and explicit result placeholders.

## Classification rule

The historical LLM output contract omitted reaction IDs already occupied by the
fixed task template. Native RL masked these IDs correctly. A pre-fix LLM campaign
is protocol-invalid whenever its task has a fixed template reaction, even if the
stored model response happened not to select a colliding ID: the model searched
and refined against the wrong action-space contract.

The quarantined LLM tasks are:

- RPA: fixed IDs 1 and 28
- Logic: fixed IDs 1, 2, 3, and 4
- Hill, ultrasensitive, and biphasic dose response: fixed ID 1
- Oscillator frequency: fixed ID 1
- Stochastic RPA: fixed IDs 1 and 57

The quarantine also contains Pro and local-model runs, pilots, aborted runs,
long-horizon and no-communication campaigns for these tasks, stale locks, and
all numerical reports, generated tables, compiled paper output, and figures that
depended on one or more invalid cohorts.

## Retained active evidence

- Every RL-only control remains active because native RL used the correct mask.
- Classifier and oscillator-mean LLM runs remain active because their templates
  contain no fixed reaction IDs. The finalized 20-seed cohorts each contain 100
  epochs, 102,300 RL evaluations, 40 or 50 LLM evaluations, request-0 HOF
  withholding, full communication, CVODE, one stable task-prompt hash, and one
  stable system-prompt hash.
- Earlier classifier/oscillator-mean pilots remain available as development
  evidence, clearly separated from the finalized 20-seed directories.
- The architecture figure, prompt appendix, and machine-readable historical
  audit remain because they document implementation and the invalidation rather
  than asserting affected outcomes.

For a homogeneous confirmatory paper cohort, classifier and oscillator mean will
still be rerun under the version-2 contract. Their retained historical cohorts
can serve as external replication checks, not as selectively reused primary
results.

## Operational state

All experiment workers, local model servers, queued campaigns, and the database
viewer were stopped before the move. The independent `dsh-harness` web service
remains running. No new experiment is scheduled or active.

The active campaign root now contains only four RL-only campaign roots and three
split Flash campaign roots containing classifier/oscillator-mean runs. The active
raw LLM root contains only the corresponding two split methods. The live paper
figure directory contains only the architecture PDF/PNG and its README.

## Paper state

Historical numerical claims were removed from the active abstract, introduction,
experiment, result, analysis, discussion, conclusion, statements, and appendix
after a forensic source copy was placed in quarantine. The active TeX contains
explicit result placeholders, and the strict build is blocked. Numerical sections
must be regenerated from contract-v2 artifacts according to
`EXPERIMENT_RECOVERY_PLAN.md`.
