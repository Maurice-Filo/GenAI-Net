# RPA Flash Harness Readiness

Checked 2026-09-17. Updated after user approval: live connectivity and the
Decider/Writer round passed. Ten of ten candidates were valid; best canonical
loss was 0.005954755380383372 without rate fitting. Elapsed time was 468.94 s.
The workspace audit confirmed zero initial HOF entries and zero exploratory
evaluations. No training or Comet upload was performed by this readiness round.

## Verified Locally

- Installed runtime: `@deepseek-ai/dsh@0.1.0-rc.8`.
- Node: `v24.19.0`, discovered automatically at
  `~/.local/nodejs/current/bin/node`; not on the interactive shell PATH.
- Existing Harness credentials file is present with mode 0600. Credential
  validity was subsequently verified by the live round; contents were not displayed or copied.
- Explicit model selection: `deepseek-official/deepseek-v4-flash`.
- The new `workspace-loss-v1` system and Decider/Writer role prompts are reused
  unchanged. The approved RPA task text is saved separately in
  `prompts/rpa-evidence-exploitation-v1.md`; no legacy prompt was replaced.
- Exactly five added reactions; forbidden IDs `[0, 1, 28]`; rates clamped to
  `[0.001, 100]`; independent-member validation; no Writer repair retries.
- The isolated first request has no HOF entries, cached HOF diagnostics,
  exploration state or previous LLM outcomes. No exploratory evaluator or
  literature tool service is supplied.
- Existing integration tests start with a populated HOF and verify withholding
  at request zero and the expected later outward-channel behavior.

Verification: 39 focused Python tests and one Node test passed (34 protocol,
client and preflight tests; three communication tests; two readiness tests).
An offline CLI preflight also passed. These are not live model results.

## Live Check

Automatic approval initially rejected the live command. After the user approved
proceeding, the scoped request was approved and completed. No bypass was used.

Executed from the repository root (use a fresh output path for another check):

```bash
.venv/bin/python -m comparisons.search_exploration.check_rpa_flash_harness \
  --output comparisons/search_exploration/reports/rpa-flash-harness-readiness-20260917 \
  --live
```

This performs one ten-candidate batch with one Decider and one Writer invocation
(each invocation may involve multiple model/tool turns), followed by canonical
CVODE evaluation. It does not launch a campaign, train RL or fit rates. It uses
the existing configured authentication without passing keys in arguments.
Existing outputs cannot be overwritten. Manifests contain prompt hashes;
call logs, validation outcomes and losses remain in the isolated workspace.

## Before the Hybrid Campaign

The frozen asynchronous runner `run_streams.py` remains unchanged. The opt-in
`run_harness_streams.py` now uses `HarnessJuliaStreamTrainer`, with separate
tests for nonblocking calls, request-zero withholding, later HOF/EIL/exclusion
feedback, candidate insertion, shared fitting admission, canonical accounting
and terminal draining. LLM outputs share the existing evaluation cap with
AdaGrad rather than receiving uncounted evaluations. Comet remains opt-in;
this readiness check itself made no uploads.
