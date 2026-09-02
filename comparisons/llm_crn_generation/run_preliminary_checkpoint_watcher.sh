#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
ANALYZER="$ROOT/comparisons/llm_crn_generation/analyze_preliminary_default_long300.py"
LOG=/local0/tmp/default-long300-checkpoint-watcher.log
TARGET="$(TZ=Europe/Zurich date -d '2026-09-01 17:30:00' +%s)"

timestamp() {
    date --iso-8601=seconds
}

now="$(date +%s)"
if (( now < TARGET )); then
    printf '[%s] Waiting until 17:30 CEST for the first checkpoint.\n' "$(timestamp)" >>"$LOG"
    sleep "$((TARGET - now))"
fi

while true; do
    printf '[%s] Checking the five-seed epoch-100 gate.\n' "$(timestamp)" >>"$LOG"
    if "$ROOT/.venv/bin/python" "$ANALYZER" >>"$LOG" 2>&1; then
        printf '[%s] Preliminary checkpoint generated successfully.\n' "$(timestamp)" >>"$LOG"
        exit 0
    fi
    printf '[%s] Gate not ready; retrying in 30 minutes.\n' "$(timestamp)" >>"$LOG"
    sleep 1800
done
