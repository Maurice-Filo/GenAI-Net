#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
WAIT_PID="${WAIT_PID:-}"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
export MPLCONFIGDIR="$ROOT/comparisons/rpa_search/.mplconfig"

if [[ -n "$WAIT_PID" ]]; then
    printf '[%s] Waiting for predecessor queue PID %s.\n' "$(date --iso-8601=seconds)" "$WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 300
    done
fi

printf '[%s] Generating final communication-ablation performance and diversity figures.\n' \
    "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" \
    "$ROOT/comparisons/rpa_search/scripts/plot_communication_ablation_over_time.py" \
    --communication-root "$CAMPAIGN_ROOT/flash-long300-20seed" \
    --isolated-root "$CAMPAIGN_ROOT/flash-no-communication-long300-20seed" \
    --isolated-status "$CAMPAIGN_ROOT/flash-no-communication-long300-20seed/status.json" \
    --output "$ROOT/comparisons/rpa_search/figures/communication_ablation_over_time.pdf"
printf '[%s] Communication-ablation figures complete.\n' "$(date --iso-8601=seconds)"
