#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
CAMPAIGN_ROOT=/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns
DSH_HOME=/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home
LAUNCHER="$ROOT/comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
ANALYZER="$ROOT/comparisons/rpa_search/scripts/analyze_initial_hof_ablation.py"
CONTROL_ID=flash-long300-20seed
CAMPAIGN_ID=flash-rpa-initial-hof-withheld-100epoch-20seed
RL_GPU_INDEX="${RL_GPU_INDEX:-1}"

SOURCE_SNAPSHOT="$CAMPAIGN_ROOT/$CAMPAIGN_ID-source.sha256"
find "$ROOT/RL4CRN" "$ROOT/comparisons/llm_crn_generation" "$ROOT/comparisons/rpa_search/src" \
    -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$SOURCE_SNAPSHOT"

RL_GPU_UUID="$(nvidia-smi --id="$RL_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"

printf '[%s] Launching 20-seed RPA context-free initial-HOF campaign.\n' "$(date --iso-8601=seconds)"
"$ROOT/.venv/bin/python" "$LAUNCHER" \
    --campaign-id "$CAMPAIGN_ID" \
    --tasks rpa \
    --seeds 20 \
    --epochs 100 \
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
    --communication-mode full \
    --withhold-initial-hof \
    --method-name genai_net_llm_flash_rpa_context_free100 \
    --run-suffix cvode_flash_rpa_context_free100 \
    --dsh-home "$DSH_HOME" \
    --workspace-root "$CAMPAIGN_ROOT" \
    --comet-project genai-net-v4-flash-paper \
    --skip-postprocessing

MPLCONFIGDIR=/local0/tmp/mpl "$ROOT/.venv/bin/python" "$ANALYZER" \
    --initial-hof-root "$CAMPAIGN_ROOT/$CONTROL_ID" \
    --context-free-root "$CAMPAIGN_ROOT/$CAMPAIGN_ID" \
    --initial-hof-status "$CAMPAIGN_ROOT/$CONTROL_ID/status.json" \
    --context-free-status "$CAMPAIGN_ROOT/$CAMPAIGN_ID/status.json" \
    --initial-hof-suffix cvode_llm_flash_long300 \
    --context-free-suffix cvode_flash_rpa_context_free100 \
    --candidate-budget 307200 \
    --max-epoch 100 \
    --tasks rpa \
    --output "$ROOT/comparisons/rpa_search/figures/initial_hof_ablation_rpa_100epoch_20seed.pdf"

printf '[%s] RPA context-free campaign and paired analysis completed.\n' "$(date --iso-8601=seconds)"
