#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
HYBRID_LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
RL_LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_breadth_rl_only_campaign.py"
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"
RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"

NO_COMM_ID=flash-no-communication-long300-20seed
LOGIC_ID=flash-logic-initial-hof-withheld-100epoch-20seed
NO_COMM_LOG=/local0/tmp/core-paper-no-communication.log
LOGIC_LOG=/local0/tmp/core-paper-logic-withheld.log

terminate_children() {
    for pid in "${NO_COMM_PID:-}" "${LOGIC_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}
trap terminate_children INT TERM

printf '[%s] Resuming the five missing no-communication runs.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$HYBRID_LAUNCHER" \
    --campaign-id "$NO_COMM_ID" \
    --tasks logic rpa \
    --seeds 20 \
    --epochs 300 \
    --rl-batch-size 1023 \
    --total-candidate-budget 307200 \
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
    --communication-mode none \
    --method-name genai_net_llm_flash_no_communication \
    --run-suffix cvode_llm_flash_no_communication \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing >"$NO_COMM_LOG" 2>&1 &
NO_COMM_PID=$!

printf '[%s] Launching 20 withheld-HOF Logic runs at ten-way concurrency.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$HYBRID_LAUNCHER" \
    --campaign-id "$LOGIC_ID" \
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
    --llm-provider deepseek-official \
    --communication-mode full \
    --withhold-initial-hof \
    --method-name genai_net_llm_flash_logic_initial_context_free100 \
    --run-suffix cvode_llm_flash_logic_initial_context_free100 \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing >"$LOGIC_LOG" 2>&1 &
LOGIC_PID=$!

set +e
wait "$NO_COMM_PID"
NO_COMM_STATUS=$?
wait "$LOGIC_PID"
LOGIC_STATUS=$?
set -e
if ((NO_COMM_STATUS != 0 || LOGIC_STATUS != 0)); then
    printf 'Core Harness campaigns failed: no-communication=%s logic-withheld=%s\n' \
        "$NO_COMM_STATUS" "$LOGIC_STATUS" >&2
    exit 1
fi

printf '[%s] Harness campaigns complete; extending five RL-only tasks to seeds 5-19.\n' \
    "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$RL_LAUNCHER" \
    --campaign-id rl-only-breadth-extension-seeds5to19 \
    --tasks dose_hill dose_ultrasensitive dose_biphasic oscillator_mean oscillator_frequency \
    --seeds 15 \
    --seed-start 5 \
    --epochs 100 \
    --batch-size 1024 \
    --candidate-budget 102400 \
    --max-parallel 15 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --method-name rl4crn_breadth \
    --run-suffix cvode_rl_only_breadth \
    --comet-project genai-net-v4-flash-paper \
    --workspace-root "$CAMPAIGN_ROOT"

printf '[%s] Extending the RL-only classifier to seeds 10-19.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$RL_LAUNCHER" \
    --campaign-id rl-only-breadth-classifier-extension-seeds10to19 \
    --tasks classifier \
    --seeds 10 \
    --seed-start 10 \
    --epochs 100 \
    --batch-size 1024 \
    --candidate-budget 102400 \
    --max-parallel 15 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --method-name rl4crn_breadth \
    --run-suffix cvode_rl_only_breadth \
    --comet-project genai-net-v4-flash-paper \
    --workspace-root "$CAMPAIGN_ROOT"

printf '[%s] Core paper completion queue finished.\n' "$(date --iso-8601=seconds)"
