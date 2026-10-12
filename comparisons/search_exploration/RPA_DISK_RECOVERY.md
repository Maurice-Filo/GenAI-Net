# RPA Disk-Full Recovery, 2026-09-16

The user authorized relocation and recovery after /local0 ran out of space.
The 90,619,869,621-byte campaign was copied to
`/local1/rossin/search-exploration/rpa-statistical-20seed-20260915` using rsync.
A checksum dry run (`rsync -acn --itemize-changes`) reported no differences
before source files were removed. The original repository path is retained as
a symlink; historical manifests, logs, failed attempts and frozen inputs remain
available at their original paths. No experimental artifact was discarded.

All 26 completed results must pass SQLite `integrity_check` and existing endpoint
validation before the campaign can restart. Completed results are not rerun.
The four interrupted attempts are GA seeds 9 and 10, Julia seed 10 and evolution
seed 10. Their logs explicitly report disk exhaustion. None has a completion
marker. Their original folders are retained, including partial results and logs.

Exact asynchronous checkpoint resumption is not supported by the existing
engine. Those same four arm-seed pairs therefore restart from the beginning in
distinct `-disk-recovery` folders. Interrupted results do not enter endpoint
comparisons and are not pooled with rerun discoveries. No best-attempt selection
is performed. Saved failed-attempt evaluation counts are reported separately as
recovery overhead and are lower bounds because final writes may have failed.
Unfinished scientific outcomes are not relabeled as successful results.

The 30 never-started jobs remain queued. All seeds, evaluator settings, reaction
constraints, rate bounds, 100-epoch hybrid horizons, batch size 1023, NN size 1024,
214793-attempt ceilings, no-timeout policy and canonical-authoritative Julia
warning policy are unchanged. Concurrency stays at two hybrids plus two GA runs,
128 allocated worker slots. No LLM calls or new Comet uploads are introduced.

Recovery uses `recover_rpa_disk.py --prepare-only`, followed by
`recover_rpa_disk.py --execute-prepared` in detached tmux. Both controller locks
are acquired, old plans/progress and source code are snapshotted, and preparation
and launch are one-shot operations. Repeated execution cannot silently overwrite
or restart an existing recovery attempt. Active plan, progress and analysis
remain accessible through the original repository path.

The campaign remains an explicitly amended workflow comparison. The disk
failure adds recovery overhead; it does not justify equal-compute claims or
selection on favorable partial outcomes.
