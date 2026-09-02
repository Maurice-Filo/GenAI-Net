#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
ANALYZER="$ROOT/comparisons/llm_crn_generation/analyze_logic_trajectory_prompt_ablation.py"
CAMPAIGN_ID=flash-logic-trajectory-prompt-100epoch-20seed
WAIT_PID="${WAIT_PID:-}"
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"

if [[ -n "$WAIT_PID" ]]; then
    printf '[%s] Waiting for campaign PID %s before Logic prompt ablation.\n' \
        "$(date --iso-8601=seconds)" "$WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 300
    done
fi

RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"

printf '[%s] Launching 20-seed Logic trajectory-prompt ablation on GPU %s.\n' \
    "$(date --iso-8601=seconds)" "$RL_GPU_UUID"
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id "$CAMPAIGN_ID" \
    --tasks logic \
    --seeds 20 \
    --epochs 100 \
    --rl-batch-size 1023 \
    --total-candidate-budget 102400 \
    --llm-candidates 10 \
    --llm-every 20 \
    --max-llm-in-flight 5 \
    --global-llm-concurrency 8 \
    --max-agent-evaluations 0 \
    --max-parallel 10 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --llm-timeout 3600 \
    --model deepseek-v4-flash \
    --task-prompt-variant logic-trajectory \
    --llm-provider deepseek-official \
    --communication-mode full \
    --withhold-initial-hof \
    --method-name genai_net_llm_flash_logic_trajectory_prompt \
    --run-suffix cvode_llm_flash_logic_trajectory_prompt \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing

printf '[%s] Regenerating the Logic prompt-conditioning report.\n' \
    "$(date --iso-8601=seconds)"
PYTHONPATH="$ROOT" MPLCONFIGDIR=/local0/tmp/mpl \
    "$ROOT/.venv/bin/python" "$ANALYZER"
