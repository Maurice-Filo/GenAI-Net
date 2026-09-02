#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
export MPLCONFIGDIR="$ROOT/comparisons/rpa_search/.mplconfig"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_breadth_rl_only_campaign.py"
ASSESSOR="$ROOT/comparisons/llm_crn_generation/assess_breadth_consistency.py"
WAIT_PID="${WAIT_PID:-}"
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"
TASKS=(dose_hill dose_ultrasensitive dose_biphasic classifier oscillator_mean oscillator_frequency)

if [[ -n "$WAIT_PID" ]]; then
    printf '[%s] Waiting for predecessor queue PID %s.\n' "$(date --iso-8601=seconds)" "$WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 300
    done
fi

RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
COMMON=(
    --epochs 100
    --batch-size 1024
    --candidate-budget 102400
    --max-parallel 15
    --cpus-per-run 4
    --rl-gpu "$RL_GPU_UUID"
    --method-name rl4crn_breadth
    --run-suffix cvode_rl_only_breadth
    --comet-project genai-net-v4-flash-paper
    --workspace-root "$CAMPAIGN_ROOT"
)

printf '[%s] Launching matched five-seed RL-only breadth cohort.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id rl-only-breadth-deterministic-5seed \
    --tasks "${TASKS[@]}" \
    --seeds 5 \
    "${COMMON[@]}"

LLM_ASSESSMENT="$CAMPAIGN_ROOT/flash-breadth-deterministic-5seed/consistency_assessment.json"
mapfile -t EXTENSION_TASKS < <(
    "$ROOT/.venv/bin/python" "$ASSESSOR" \
        --raw-root "$ROOT/comparisons/rpa_search/data/raw" \
        --output "$LLM_ASSESSMENT" \
        --tasks-only
)
if ((${#EXTENSION_TASKS[@]})); then
    printf '[%s] Mirroring LLM-arm seed extensions in RL-only baseline: %s.\n' \
        "$(date --iso-8601=seconds)" "${EXTENSION_TASKS[*]}"
    "$ROOT/.venv/bin/python" "$LAUNCHER" \
        --campaign-id rl-only-breadth-deterministic-extension-seeds5to9 \
        --tasks "${EXTENSION_TASKS[@]}" \
        --seeds 5 \
        --seed-start 5 \
        "${COMMON[@]}"
else
    printf '[%s] No paired breadth extensions required.\n' "$(date --iso-8601=seconds)"
fi

printf '[%s] Matched RL-only breadth cohort completed.\n' "$(date --iso-8601=seconds)"
