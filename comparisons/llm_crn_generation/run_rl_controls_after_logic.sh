#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
RAW="$ROOT/comparisons/rpa_search/data/raw"
METHOD=genai_net_llm_flash_logic_initial_context_free100
SUFFIX=cvode_llm_flash_logic_initial_context_free100
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_breadth_rl_only_campaign.py"
RL_GPU_UUID="$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader | tr -d ' ')"

complete_logic_runs() {
    local count=0
    local seed
    for seed in $(seq 0 19); do
        if [[ -f "$RAW/$METHOD/logic_full102400_seed${seed}_${SUFFIX}/completed.json" ]]; then
            count=$((count + 1))
        fi
    done
    printf '%s' "$count"
}

while [[ "$(complete_logic_runs)" -lt 20 ]]; do
    sleep 60
done

printf '[%s] Logic withholding complete; filling ten freed CVODE slots with RL controls.\n' \
    "$(date --iso-8601=seconds)"

"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id rl-only-breadth-extension-seeds5to19 \
    --tasks dose_hill dose_ultrasensitive dose_biphasic oscillator_mean oscillator_frequency \
    --seeds 15 \
    --seed-start 5 \
    --epochs 100 \
    --batch-size 1024 \
    --candidate-budget 102400 \
    --max-parallel 14 \
    --global-slots 15 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --method-name rl4crn_breadth \
    --run-suffix cvode_rl_only_breadth \
    --comet-project genai-net-v4-flash-paper \
    --workspace-root "$CAMPAIGN_ROOT"

"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id rl-only-breadth-classifier-extension-seeds10to19 \
    --tasks classifier \
    --seeds 10 \
    --seed-start 10 \
    --epochs 100 \
    --batch-size 1024 \
    --candidate-budget 102400 \
    --max-parallel 14 \
    --global-slots 15 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --method-name rl4crn_breadth \
    --run-suffix cvode_rl_only_breadth \
    --comet-project genai-net-v4-flash-paper \
    --workspace-root "$CAMPAIGN_ROOT"

printf '[%s] Accelerated RL-control handoff complete.\n' "$(date --iso-8601=seconds)"
