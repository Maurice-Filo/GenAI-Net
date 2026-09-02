#!/usr/bin/env bash

set -euo pipefail

# Keep nvidia-smi and CUDA numeric indices aligned for the llama.cpp GPU setting.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
LOCAL_LLM_DIR=/local0/home/rossin/local-llm
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
LOCAL_URL=http://127.0.0.1:8080/v1
CURRENT_STATUS=(
  "$CAMPAIGN_ROOT/20260821T-genai-net-llm-pro-paper-20seed-r3/status.json"
  "$CAMPAIGN_ROOT/20260821T-genai-net-llm-pro-logic-20seed-timeout3600-r4/status.json"
)
RL_GPU_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
SERVER_LAUNCHER_PID=

timestamp() {
    date --iso-8601=seconds
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

campaigns_are_idle() {
    local status
    for status in "${CURRENT_STATUS[@]}"; do
        [[ -f "$status" ]] || return 1
        grep -q '^  "active": \[\],$' "$status" || return 1
        grep -q '^  "pending": \[\],$' "$status" || return 1
    done
}

wait_for_gpu_memory() {
    local minimum_free_mib="$1"
    local free_mib
    while true; do
        free_mib="$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= minimum_free_mib )); then
            log "GPU 1 has ${free_mib} MiB free; continuing."
            return 0
        fi
        log "Waiting for GPU 1: ${free_mib:-unknown} MiB free, ${minimum_free_mib} MiB required."
        sleep 300
    done
}

stop_server() {
    if [[ -s "$LOCAL_LLM_DIR/run/llama-server.pid" ]]; then
        "$LOCAL_LLM_DIR/stop-llm.sh" || true
    fi
    if [[ -n "$SERVER_LAUNCHER_PID" ]]; then
        wait "$SERVER_LAUNCHER_PID" 2>/dev/null || true
        SERVER_LAUNCHER_PID=
    fi
}

start_server() {
    local serve_script="$1"
    log "Starting $serve_script; first use may download its pinned GGUF."
    "$LOCAL_LLM_DIR/$serve_script" &
    SERVER_LAUNCHER_PID=$!
    for _ in $(seq 1 1440); do
        if curl --silent --fail --max-time 2 http://127.0.0.1:8080/health >/dev/null; then
            log "Local model endpoint is healthy."
            return 0
        fi
        if ! kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; then
            wait "$SERVER_LAUNCHER_PID"
            return 1
        fi
        sleep 10
    done
    log "Local model did not become healthy within four hours."
    return 1
}

smoke_model() {
    local model="$1"
    local smoke_root="$CAMPAIGN_ROOT/local-qwen-smoke-$model-$(date -u +%Y%m%dT%H%M%SZ)"
    log "Running Harness smoke test for $model on Logic and RPA."
    "$ROOT/.venv/bin/python" \
        "$ROOT/comparisons/llm_crn_generation/run_mmc2_harness_smoke.py" \
        --task all \
        --num-candidates 1 \
        --model "$model" \
        --llm-provider local-llama \
        --llm-base-url "$LOCAL_URL" \
        --timeout 300 \
        --dsh-home "$DSH_HOME" \
        --workspace-root "$smoke_root"
}

run_campaign() {
    local model="$1"
    local method="$2"
    local suffix="$3"
    local campaign_id="local-$model-20seed"
    log "Launching $campaign_id (20 Logic + 20 RPA seeds)."
    "$ROOT/.venv/bin/python" \
        "$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py" \
        --campaign-id "$campaign_id" \
        --tasks logic rpa \
        --max-parallel 1 \
        --global-llm-concurrency 1 \
        --cpus-per-run 4 \
        --rl-gpu "$RL_GPU_UUID" \
        --llm-timeout 3600 \
        --model "$model" \
        --llm-provider local-llama \
        --llm-base-url "$LOCAL_URL" \
        --method-name "$method" \
        --run-suffix "$suffix" \
        --dsh-home "$DSH_HOME" \
        --workspace-root "$CAMPAIGN_ROOT" \
        --comet-project genai-net-local-llm-paper
}

trap stop_server EXIT INT TERM

log "Queue active; waiting for the current Pro campaigns to have no active or pending runs."
until campaigns_are_idle; do
    sleep 300
done

log "Current campaigns are idle. Building pinned llama.cpp without sudo."
"$LOCAL_LLM_DIR/build.sh"

QWEN9_SMOKE_PASSED=0
QWEN27_SMOKE_PASSED=0

if [[ "${SKIP_QWEN9_SMOKE:-0}" == 1 ]]; then
    log "Skipping repeated 9B smoke; the bounded test already failed agent completion."
else
    start_server serve-qwen35-9b.sh
    if smoke_model qwen35-9b; then
        QWEN9_SMOKE_PASSED=1
    else
        log "9B smoke failed; its full campaign is disabled."
    fi
    stop_server
fi

wait_for_gpu_memory 22500
start_server serve-qwen38-27b.sh
if smoke_model qwen38-27b; then
    QWEN27_SMOKE_PASSED=1
else
    log "27B smoke failed; its full campaign is disabled."
fi
stop_server

if (( QWEN9_SMOKE_PASSED )); then
    start_server serve-qwen35-9b.sh
    run_campaign qwen35-9b genai_net_llm_qwen35_9b cvode_llm_qwen35_9b
    stop_server
fi

if (( QWEN27_SMOKE_PASSED )); then
    start_server serve-qwen38-27b.sh
    run_campaign qwen38-27b genai_net_llm_qwen38_27b cvode_llm_qwen38_27b
    stop_server
fi

log "Local-model queue complete (9B smoke=$QWEN9_SMOKE_PASSED, 27B smoke=$QWEN27_SMOKE_PASSED)."
