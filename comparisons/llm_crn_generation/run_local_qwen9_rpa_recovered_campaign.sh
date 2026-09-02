#!/usr/bin/env bash

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
LOCAL_LLM_DIR=/local0/home/rossin/local-llm
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
CAMPAIGN_ID=local-qwen35-9b-rpa-default100-recovered-20seed-r3
LOCAL_URL=http://127.0.0.1:8080/v1
RL_GPU_UUID=GPU-e28480e9-6ce2-60d6-887a-7c2a0cbd1e19

timestamp() {
    date --iso-8601=seconds
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

stop_server() {
    "$LOCAL_LLM_DIR/stop-llm.sh" || true
}

trap stop_server EXIT

if ! curl --silent --fail --max-time 5 http://127.0.0.1:8080/health >/dev/null; then
    log "The localhost Qwen server is not healthy; refusing to launch the campaign."
    exit 1
fi

log "Launching the recover-valid-subset Qwen3.5-9B RPA campaign with two RL workers."
"$ROOT/.venv/bin/python" \
    "$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py" \
    --campaign-id "$CAMPAIGN_ID" \
    --tasks rpa \
    --seeds 20 \
    --seed-start 0 \
    --epochs 100 \
    --rl-batch-size 1023 \
    --total-candidate-budget 102400 \
    --llm-candidates 10 \
    --llm-every 20 \
    --max-llm-in-flight 5 \
    --max-parallel 2 \
    --global-llm-concurrency 1 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --llm-timeout 1800 \
    --max-agent-evaluations 0 \
    --model qwen35-9b \
    --generation-backend harness \
    --llm-provider local-llama \
    --llm-base-url "$LOCAL_URL" \
    --communication-mode full \
    --withhold-initial-hof \
    --recover-valid-llm-candidates \
    --method-name genai_net_llm_qwen35_9b_rpa_default100_recovered \
    --run-suffix cvode_llm_qwen35_9b_rpa_default100_recovered \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-local-llm-paper \
    --skip-postprocessing

log "Recover-valid-subset Qwen3.5-9B RPA campaign completed."
