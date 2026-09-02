#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
CAMPAIGN_ID=flash-breadth-initial-hof-withheld-100epoch-20seed
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"
TASKS=(
    dose_hill
    dose_ultrasensitive
    dose_biphasic
    classifier
    oscillator_mean
    oscillator_frequency
)

RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"

printf '[%s] Launching 120-run initial-HOF-withheld breadth campaign.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id "$CAMPAIGN_ID" \
    --tasks "${TASKS[@]}" \
    --seeds 20 \
    --epochs 100 \
    --rl-batch-size 1023 \
    --total-candidate-budget 102400 \
    --llm-candidates 10 \
    --llm-every 20 \
    --max-llm-in-flight 5 \
    --global-llm-concurrency 8 \
    --max-agent-evaluations 0 \
    --max-parallel 15 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --llm-timeout 3600 \
    --model deepseek-v4-flash \
    --llm-provider deepseek-official \
    --communication-mode full \
    --withhold-initial-hof \
    --method-name genai_net_llm_flash_breadth_initial_context_free20 \
    --run-suffix cvode_llm_flash_breadth_initial_context_free20 \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing

printf '[%s] Initial-HOF-withheld breadth campaign completed.\n' "$(date --iso-8601=seconds)"
