#!/usr/bin/env bash

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
CAMPAIGN_ID=flash-default-initial-hof-withheld-long300-20seed
RL_GPU_UUID=GPU-e28480e9-6ce2-60d6-887a-7c2a0cbd1e19

timestamp() {
    date --iso-8601=seconds
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

log "Launching the clean 300-epoch default full-duplex Flash campaign."
"$ROOT/.venv/bin/python" \
    "$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py" \
    --campaign-id "$CAMPAIGN_ID" \
    --tasks logic rpa \
    --seeds 20 \
    --seed-start 0 \
    --epochs 300 \
    --rl-batch-size 1023 \
    --total-candidate-budget 307200 \
    --llm-candidates 10 \
    --llm-every 20 \
    --max-llm-in-flight 5 \
    --max-parallel 10 \
    --global-llm-concurrency 8 \
    --cpus-per-run 4 \
    --rl-gpu "$RL_GPU_UUID" \
    --llm-timeout 3600 \
    --max-agent-evaluations 0 \
    --model deepseek-v4-flash \
    --generation-backend harness \
    --llm-provider deepseek-official \
    --communication-mode full \
    --withhold-initial-hof \
    --method-name genai_net_llm_flash_default_long300 \
    --run-suffix cvode_llm_flash_default_long300 \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-llm-paper \
    --skip-postprocessing

log "Clean 300-epoch default full-duplex Flash campaign completed."
